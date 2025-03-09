"""
title: Smart Document Generator
author: Unknown
author_url: Unknown
git_url: https://github.com/gpillon/document_generator_pipe
description: Smart Document Generator with section editing and dynamic action detection
required_open_webui_version: 0.4.3
requirements: fastapi, pydantic, httpx
version: 0.5.0
licence: MIT
"""

# TODO(feature): Add a "create_summarization" action that creates a summarization of the document

# TODO(feature): add a "cleanup" action that cleans up the document, checks, the language style, etc...
# TODO(feature): idea for editing document.. each time that the the pipe is called, we can compare the document with the original document, and detect the changes.
# TODO(feature): split list template in "detail template" adn "list only template names"
# TODO(feature): support multiple actions; actions can be queued, to do many things toghether.
# TODO(feature): add RETRIES MAYBE WITH "LIMITED" EXPONENTIAL BACKOPP TO API CALLS

# TODO(fix): fix templates retrival from GIT
# TODO(fix, verify): self.document = loaded_document not working in document in "memory" (file working)
# TODO(fix): Remove section is not smart wants always "the section name" and not enterpretes the "section title"
# TODO(fix,maybe fixed): when adding subsection the title contains the section title and the subsection title

# TODO(enhancement): remove "smart" model and combine with "if user prompt start with [pattern] then do [action]"
# TODO(enhancement): Requesting "delicate" operations like section removal, cuold be done asking confirm to the user, using messages["user"][0] and mesages["user"][1]
# TODO(enhancement): on wrie_full_doc, the user could want to generate also the template; so with a flag like "generate_template", we could achieve this. 

# TODO(enhancement): refactor, putting "client" in the context

# TODO(fix, URGENT): Edit document is currently not working, messes up the whole structure, need to fix

import json
import httpx
import asyncio
import logging
import time
import re
import os
import subprocess
import shutil
import glob
from pydantic import BaseModel, Field
from typing import Union, Generator, Iterator

# RAG imports
from open_webui.retrieval.utils import (
    query_collection,
    query_collection_with_hybrid_search,
)
from open_webui.retrieval.web.utils import get_web_loader
from open_webui.retrieval.web.duckduckgo import search_duckduckgo


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipe:
    class Valves(BaseModel):
        API_HOST: str = Field(
            default="https://api.openai.com/v1",
            description="API Host URL for content generation"
        )
        API_KEY: str = Field(
            default="",
            description="API Key for the model provider"
        )
        MODEL_ID: str = Field(
            default="gpt-3.5-turbo",
            description="Model ID to use for generation (comma-separated for multiple models)"
        )
        MAX_TOKENS: int = Field(
            default=40000,
            description="Maximum tokens per section generation"
        )
        THINKING_API_HOST: str = Field(
            default="https://api.openai.com/v1",
            description="API Host URL for the thinking model"
        )
        THINKING_API_KEY: str = Field(
            default="",
            description="API Key for the thinking model (defaults to API_KEY if empty)"
        )
        THINKING_MODEL_ID: str = Field(
            default="gpt-4o",
            description="Model ID to use for thinking/action detection"
        )
        MAX_THINKING_TOKENS: int = Field(
            default=40000,
            description="Maximum tokens for thinking"
        )
        SMART_MODEL: bool = Field(
            default=True,
            description="Use the thinking model for action detection (true) or use hardcoded patterns (false)"
        )
        TEMPERATURE: float = Field(
            default=0.7,
            description="Temperature for generation"
        )
        TIMEOUT: int = Field(
            default=300,
            description="Timeout in seconds for API requests"
        )
        REQUEST_TIMEOUT: float = Field(
            default=60.0,
            description="Timeout for individual request chunks"
        )
        DOCUMENT_STORAGE: str = Field(
            default="memory",
            description="Storage mechanism for documents (memory or file)"
        )
        TEMPLATE_NAME: str = Field(
            default="basic",
            description="The fallback template to use for document generation if no template is specified"
        )
        GIT_TEMPLATE_REPOS: str = Field(
            default="",
            description="Comma-separated list of Git repositories containing additional templates"
        )
        ENABLE_RAG: bool = Field(
            default=True,
            description="Enable Retrieval Augmented Generation when creating document sections"
        )
        RELEVANCE_THRESHOLD: float = Field(
            default=0.3,
            description="Default relevance threshold to use for RAG (if empty, will use default value)"
        )
        RAG_RESULT_COUNT: int = Field(
            default=5,
            description="Number of documents to retrieve for RAG"
        )
        RAG_WEB_SEARCH: bool = Field(
            default=False,
            description="Enable web search retrieval for RAG"
        )
        RAG_WEB_SEARCH_ENGINE: str = Field(
            default="duckduckgo",
            description="Web search engine to use for RAG"
        )
        RAG_HYBRID_SEARCH: bool = Field(
            default=True,
            description="Use hybrid search (vector + reranking) for better results"
        )

        class Config:
            json_schema_extra = {
                "examples": [
                    {
                        "TEMPLATE_NAME": "technical_documentation",
                        "GIT_TEMPLATE_REPOS": "https://github.com/user/doc-templates.git,https://github.com/organization/templates-collection.git"
                    }
                ]
            }

    def __init__(self):
        self.valves = self.Valves()
        # In-memory store for documents
        self.documents = {}
        # New flag: use the thinking model also for content generation
        self.use_thinking_for_generation = True

        self.loaded_templates = False

        # Create necessary directories
        os.makedirs("/app/backend/data/doc_generator_pipe", exist_ok=True)
        os.makedirs("/app/backend/data/doc_generator_pipe/git", exist_ok=True)
        os.makedirs("/app/backend/data/doc_generator_pipe/templates", exist_ok=True)

        # Embedded templates with support for nested sections and section types
        self.templates = {
            "basic": {
                "name": "Basic Document",
                "description": "A simple document with introduction, body, and conclusion",
                "sections": [
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an introduction about {topic}. Provide context and background.",
                        "sectionType": "free_text",
                        "children": []  # No children for this section
                    },
                    {
                        "name": "body",
                        "title": "Main Content",
                        "prompt": "Provide detailed information about {topic}. Include key facts, analysis, and insights.",
                        "sectionType": "free_text",
                        "children": [
                            {
                                "name": "body_part1",
                                "title": "Key Concepts",
                                "prompt": "Explain the key concepts related to {topic}.",
                                "sectionType": "free_text",
                                "children": []
                            },
                            {
                                "name": "body_part2",
                                "title": "Analysis",
                                "prompt": "Provide analysis and interpretation of {topic}.",
                                "sectionType": "free_text",
                                "children": []
                            }
                        ]
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion summarizing the key points about {topic}.",
                        "sectionType": "free_text",
                        "children": []
                    }
                ]
            },
            "research": {
                "name": "Research Paper",
                "description": "Academic research paper structure",
                "sections": [
                    {
                        "name": "abstract",
                        "title": "Abstract",
                        "prompt": "Write a concise abstract for a research paper about {topic}.",
                        "sectionType": "free_text"
                    },
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an academic introduction for research on {topic}. Include research questions.",
                        "sectionType": "free_text"
                    },
                    {
                        "name": "literature",
                        "title": "Literature Review",
                        "prompt": "Provide a literature review for research on {topic}. Reference key studies and findings.",
                        "sectionType": "free_text"
                    },
                    {
                        "name": "methodology",
                        "title": "Methodology",
                        "prompt": "Describe a research methodology for studying {topic}.",
                        "sectionType": "free_text"
                    },
                    {
                        "name": "results",
                        "title": "Results and Discussion",
                        "prompt": "Present hypothetical results and discussion for research on {topic}.",
                        "sectionType": "free_text"
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion for academic research on {topic}. Include limitations and future directions.",
                        "sectionType": "free_text"
                    },
                    {
                        "name": "references",
                        "title": "References",
                        "prompt": "Generate a sample reference list for research on {topic}. Use proper academic citation format.",
                        "sectionType": "free_text"
                    }
                ]
            },
            "advanced": {
                "name": "Advanced Document with Special Sections",
                "description": "Document with fixed, template, and generated sections",
                "sections": [
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an introduction about {topic}. Provide context and background.",
                        "sectionType": "free_text",
                        "children": []
                    },
                    {
                        "name": "toc",
                        "title": "Table of Contents",
                        "prompt": "Generate a table of contents for the document.",
                        "sectionType": "toc",
                        "children": []
                    },
                    {
                        "name": "non_tech_stuff",
                        "title": "Non-Technical Stuff",
                        "prompt": "Write non-technical stuff about {topic}.",
                        "sectionType": "empty",
                        "children": [{
                            "name": "disclaimer",
                            "title": "Disclaimer",
                            "sectionType": "fixed",
                            "template_text": "This document is for informational purposes only. The information provided is not legal or professional advice and should not be relied upon as such. Consult with appropriate experts before making decisions based on this content.",
                            "children": [{
                                "name": "report_info",
                                "title": "Report Information",
                                "sectionType": "populate",
                                "template_text": "Document ID: {doc_id}\nDate Generated: {date}\nAuthor: {author}\nVersion: {version}",
                                "populateFields": [
                                    {"name": "doc_id", "prompt": "Generate a unique document ID for this report"},
                                    {"name": "date", "prompt": "Generate today's date in YYYY-MM-DD format"},
                                    {"name": "author", "prompt": "Generate an appropriate author name for a report on {topic}"},
                                    {"name": "version", "prompt": "Generate a version number for this document"}
                                ],
                                "children": []
                            }]
                        }]
                    },
                    {
                        "name": "main_content",
                        "title": "Main Content",
                        "prompt": "Write detailed content about {topic}.",
                        "sectionType": "free_text",
                        "children": []
                    },
                    {
                        "name": "copyright",
                        "title": "Copyright Notice",
                        "sectionType": "fixed",
                        "template_text": "© 2023 Document Generator. All rights reserved. No part of this document may be reproduced without permission.",
                        "children": []
                    }
                ]
            },
            "custom": {
                "name": "Custom Document",
                "description": "User-defined document structure",
                "sections": [
                     {
                        "name": "sample_section",
                        "title": "Sample Section",
                        "prompt": "Generate a sample section for a document on {topic}.",
                        "sectionType": "free_text"
                    }
                ]
            }
        }
        
        # Load templates from Git repositories and local storage
        # self.load_git_templates()
        # self._load_local_templates()

        logger.info(f"Templates loaded: {self.templates}")

        self.system_prompts = {
            "generic_init": (
                "You are an AI assistant helping with document generation. The document is in markdown format. Do not add comments or opinions. "
            ),
            "no_sections": (
                "Do not include any additional sections marked by the '#' symbol, headings, or subsections. "
            ),
            "no_conclusions": (
                "The response must not contain phrases like 'finally', 'in conclusion', 'in summary', or 'in the end'. "
            ),
            "rag": (
                "You are an AI assistant helping with document generation. You are given a topic and a prompt. You must generate a query for getting data from a Vector Database. You must give only the query, nothing else. You MUST not add comments, just the query." + 
                "for example, if the prompt is 'generate  a section for the topic: 'capital of France',' and the topic is 'France', the query for getting data from a Vector Database should be 'What is the capital of France?'"
            )
        }

        # Define available actions
        self.document_metadata = [
            {"name": "topic", "type": "string", "description": "The topic of the document"},
            {"name": "title", "type": "string", "description": "The title of the document"},
            {"name": "template_id", "type": "enum", "values": list(self.templates.keys()), "description": "The template to use for document generation"},
            {"name": "language", "type": "string", "description": "The language of the document"},
            {"name": "style", "type": "string", "description": "The style of the document"},
            {"name": "tone", "type": "string", "description": "The tone of the document"},
            {"name": "length", "type": "string", "description": "The length of the document"}
        ]
        
        self.actions = {
            "write_full_doc": {
                "description": "Generate a complete document",
                "patterns": ["create", "free_text", "make a document", "prepare a document"],
                "requires": [] + self.document_metadata
            },
            "edit_section": {
                "description": "Edit or regenerate a specific section",
                "patterns": ["edit", "change", "modify", "update", "revise"],
                "requires": [
                    {"name": "section_name", "type": "string", "description": "The name of the section to edit"},
                    {"name": "instructions", "type": "string", "description": "Editing instructions for the section"}
                ]
            },
            "add_section": {
                "description": "Add a new section or sub-section to the document",
                "patterns": ["add", "insert", "include", "create section"],
                "requires": [
                    {"name": "section_title", "type": "string", "description": "The title for the new section. NEVER  include: formatting, markdown, symbols, paragraph numbers or other section titles."},
                    {"name": "instructions", "type": "string", "description": "Instructions for the section content"}
                ]
            },
            "remove_section": {
                "description": "Remove a section from the document",
                "requires": [
                    {"name": "section_name", "type": "string", "description": "The name of the section to remove"}
                ]
            },
            "summarize_doc": {
                "description": "Summarize the existing document",
                "patterns": ["summarize", "summary", "brief overview"],
                "requires": []
            },
            # "expand_section": {
            #     "description": "Expand a specific section with more details",
            #     "patterns": ["expand", "elaborate", "more details", "extend"],
            #     "requires": [
            #         {"name": "section_name", "type": "string", "description": "The name of the section to expand"},
            #         {"name": "instructions", "type": "string", "description": "Instructions for expanding the section"}
            #     ]
            # },
            "rewrite_section": {
                "description": "Rewrite a section in a different style",
                "patterns": ["rewrite", "rephrase", "different style", "new style"],
                "requires": [
                    {"name": "section_name", "type": "string", "description": "The name of the section to rewrite"},
                    {"name": "style", "type": "string", "description": "The style to rewrite the section in"}
                ]
            },
            "list_sections": {
                "description": "List all sections in the current document",
                "patterns": ["list sections", "show sections", "what sections"],
                "requires": []
            },
            "generate_tags": {
                "description": "Generate tags for the chat content",
                "patterns": ["generate 1-3 broad tags categorizing the main themes of the chat history"],
                "requires": []
            },
            "list_actions": {
                "description": "List all available actions with their descriptions",
                "patterns": ["list actions", "show actions", "what actions"],
                "requires": []
            },
            "get_all_templates": {
                "description": "Display all available document templates",
                "patterns": ["show templates", "list templates", "available templates", "get templates", "what templates"],
                "requires": []
            },
            "generate_template": {
                "description": "Generate a document template from a markdown document",
                "patterns": ["generate template", "create template", "make template", "extract template"],
                "requires": [
                    {"name": "markdown", "type": "string", "description": "Markdown document text to extract template from"}
                ]
            },
            "print_document": {
                "description": "Print the current document in its entirety.",
                "patterns": [
                    r"print(?:\s+the)?(?:\s+current)?(?:\s+document)",
                    r"show(?:\s+the)?(?:\s+(?:full|entire|complete|current))?(?:\s+document)",
                    r"display(?:\s+the)?(?:\s+(?:full|entire|complete|current))?(?:\s+document)"
                ],
                "requires": []
            },
            "delete_template": {
                "description": "Delete a document template",
                "patterns": ["delete template", "remove template", "remove document template"],
                "requires": [
                    {"name": "template_id", "type": "string", "description": "The ID of the template to delete"}
                ]
            },
            "get_template_json": {
                "description": "Get the JSON structure of a document template",
                "patterns": ["get template", "get document template", "get template json", "get document template json"],
                "requires": [
                    {"name": "template_id", "type": "string", "description": "The ID of the template to retrieve"}
                ]
            },
            "update_template": {
                "description": "Update a template with the provided JSON",
                "patterns": ["update template", "update document template", "update template json", "update document template json"],
                "requires": [
                    {"name": "template_id", "type": "string", "description": "The ID of the template to update"},
                    {"name": "template_json", "type": "string", "description": "The JSON structure for the template"}
                ]
            },
            "import_document": {
                "description": "Import a document from text pasted by the user",
                "patterns": ["import document", "import text", "import markdown"],
                "requires": [
                    {"name": "document_text", "type": "string", "description": "Markdown document text to import"}
                ]
            },
        }

    def pipes(self):
        """Return available pipe options based on model IDs."""
        models = self.valves.MODEL_ID.split(",")
        options = []
        for model in models:
            model_id = model.strip()
            options.append({
                "id": f"{model_id}:Document Generator",
                "name": f"{model_id} - Document Generator",
            })
        return options

    async def pipe(self, body: dict, __event_emitter__=None, __metadata__=None, __request__=None, __user__=None, __task__=None, __task_body__=None, __files__=None) -> Union[str, Generator, Iterator]:
        """
        Main pipeline handler that dispatches to specific action handlers.
        It extracts user messages, detects the action to perform, and then routes the request accordingly.
        """

        if not self.loaded_templates:
            self.load_git_templates()
            self._load_local_templates()
            self.loaded_templates = True

        user_message = ""
        model_id = ""
        messages = []
        chat_id = __metadata__.get("chat_id", "default_chat")
        request = __request__
        user = __user__
        collection_names = self.extract_collection_names(__metadata__) 

        ctx = {
            "chat_id": chat_id,
            "request": request,
            "user": user,
            "task": __task__,
            "task_body": __task_body__,
            "files": __files__,
            "metadata": __metadata__,
            "collection_names": collection_names,
            "event_emitter": __event_emitter__,
            "result": "",
            "body": body,
            "generator": {
                "headers": self._get_api_headers(self.valves.API_KEY),
                "model_id": self.valves.MODEL_ID,
            },
            "thinker": {
                "headers": self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY),
                "model_id": self.valves.THINKING_MODEL_ID,
            },
            "rag": {
                "use_rag": self.valves.ENABLE_RAG and len(collection_names) > 0,
            }
        }       

        # Extract user message from the last user entry
        if "messages" in body and body["messages"]:
            messages = body["messages"]
            last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user_message:
                user_message = last_user_message.get("content", "")

        # Extract model ID from body
        if "model" in body:
            model_id = body["model"]
            if "." in model_id:
                model_id = model_id.split(".")[-1]
            ctx["generator"]["model_id"] = model_id

        # If using file storage, try to load an existing document
        ctx["document"] = None
        if self.valves.DOCUMENT_STORAGE == "file":
            loaded_document = self.load_document(chat_id)
            if loaded_document:
                self.documents[chat_id] = loaded_document
                self.document = loaded_document
                logger.info(f"Loaded existing document for chat {chat_id}")
                ctx["document"] = loaded_document

        # Validate API key
        if not self.valves.API_KEY:
            if ctx["event_emitter"]:
                await ctx["event_emitter"]({
                    "type": "status",
                    "data": {"description": "Error: API key not configured", "done": True}
                })
            yield "Error: API key not configured. Please set the API_KEY valve."
            return

        try:
            # Start thinking process
            yield "<think>\n"

            # # Check if the previous message was edited and contains a document
            # if ctx["document"]:
            #     edited_document = await self.check_for_document_edits(messages, chat_id, ctx)
            #     if edited_document:
            #         yield f"Detected document edits. Document has been updated.\n\n"
        
            yield "🧠 Analyzing your request...\n"

            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            timeout = httpx.Timeout(self.valves.TIMEOUT, connect=5.0)
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                # Detect the action
                ctx["client"] = client
                action_results = await self.detect_action(client, user_message, ctx)
                if not action_results:
                    yield "No actions detected. Falling back to default document generation...\n"
                    action_results = [{"action": "write_full_doc", "parameters": {"topic": user_message, "template_id": self.valves.TEMPLATE_NAME}}]
                
                yield f"Detected {len(action_results)} action(s):\n"
                for i, action_result in enumerate(action_results):
                    action = action_result["action"]
                    params = action_result.get("parameters", {})
                    confidence = action_result.get("confidence", 0.0)
                    yield f"{i+1}. {self.actions[action]['description']} ({action}) (confidence: {confidence:.2f})\n"
                    yield f"   Parameters: {json.dumps(params, indent=2)}\n\n"

                # Execute each action
                for i, action_result in enumerate(action_results):
                    action = action_result["action"]
                    params = action_result.get("parameters", {})
                    
                    yield f"Executing action {i+1}/{len(action_results)}: {action}\n"
                    if i > 0:
                        ctx["result"] += "\n---\n\n"
                    
                    # Dispatch to appropriate handler
                    if action == "generate_tags":
                        result = await self.handle_generate_tags(client, user_message, ctx)
                        logger.info(f"Generated tags: '{result}'")
                        yield result
                    elif action == "write_full_doc":
                        async for chunk in self.handle_write_full_doc(client, chat_id, user_message, model_id, params, ctx):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "edit_section":
                        async for chunk in self.handle_edit_section(client, chat_id, params, ctx):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "add_section":
                        async for chunk in self.handle_add_section(client, chat_id, user_message, params, ctx):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "remove_section":
                        async for chunk in self.handle_remove_section(chat_id, params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "list_sections":
                        async for chunk in self.handle_list_sections(chat_id):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "summarize_doc":
                        async for chunk in self.handle_summarize_doc(client, chat_id, params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "expand_section":
                        async for chunk in self.handle_expand_section(client, chat_id, params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "rewrite_section":
                        async for chunk in self.handle_rewrite_section(client, chat_id, params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "list_actions":
                        async for chunk in self.handle_list_actions():
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "get_all_templates":
                        async for chunk in self.handle_get_all_templates():
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "generate_template":
                        async for chunk in self.generate_template_from_markdown(client, user_message):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "print_document":
                        async for chunk in self.handle_print_document(chat_id):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "delete_template":
                        async for chunk in self.handle_delete_template(params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "get_template_json":
                        async for chunk in self.handle_get_template_json(params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "update_template":
                        async for chunk in self.handle_update_template(params):
                            yield self.handle_action_chunk(chunk, ctx)
                    elif action == "import_document":
                        async for chunk in self.handle_import_document(chat_id, params, ctx):
                            yield self.handle_action_chunk(chunk, ctx)
                    else:
                        yield f"Action '{action}' not implemented or recognized.\n"
                        
                    # Add a separator between actions if there are multiple
                    if i < len(action_results) - 1:
                        yield "\n---\n\n"

                yield "</think>\n\n"
                yield ctx["result"]

            if ctx["event_emitter"]:
                await ctx["event_emitter"]({
                    "type": "status",
                    "data": {"description": "Operation completed", "done": True}
                })
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Document operation error: {error_msg}")
            if ctx["event_emitter"]:
                await ctx["event_emitter"]({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            yield "</think>\n\n"

    async def detect_action(self, client, user_message, ctx):
        """
        Detect the action(s) requested by the user.
        Returns a list of dicts with keys: action, parameters, and confidence.
        """
        lower_msg = user_message.lower()
        logger.info(f"Detecting action for message: {user_message}")
        if user_message.strip().lower().startswith("import document") or user_message.strip().lower().startswith("import_document"):
            return [{"action": "import_document", "parameters": {"document_text": user_message.strip()}, "confidence": 1.0}]

        if "generate 1-3 broad tags categorizing the main themes of the chat history" in user_message:
            return [{"action": "generate_tags", "parameters": {}, "confidence": 1.0}]

        help_patterns = self.actions["list_actions"]["patterns"]
        if any(pattern in lower_msg for pattern in help_patterns):
            return [{"action": "list_actions", "parameters": {}, "confidence": 1.0}]

        template_patterns = self.actions["get_all_templates"]["patterns"]
        if any(pattern in lower_msg for pattern in template_patterns):
            return [{"action": "get_all_templates", "parameters": {}, "confidence": 1.0}]

        template_gen_patterns = self.actions["generate_template"]["patterns"]
        if any(pattern in lower_msg for pattern in template_gen_patterns) and '#' in user_message:
            return [{"action": "generate_template", "parameters": {"markdown": user_message}, "confidence": 0.9}]

        if not self.valves.SMART_MODEL:
            actions = []
            if any(p in lower_msg for p in self.actions["write_full_doc"]["patterns"]):
                template_id = self.valves.TEMPLATE_NAME
                for tid in self.templates.keys():
                    if tid.lower() in lower_msg:
                        template_id = tid
                        break
                actions.append({"action": "write_full_doc", "parameters": {"topic": user_message, "template_id": template_id}, "confidence": 0.8})
            if any(p in lower_msg for p in self.actions["edit_section"]["patterns"]):
                match = re.search(r'(edit|change|modify|update|revise)\s+(\w+)\s+section', lower_msg)
                section_name = match.group(2) if match else "introduction"
                actions.append({"action": "edit_section", "parameters": {"section_name": section_name, "instructions": user_message}, "confidence": 0.7})
            if any(p in lower_msg for p in self.actions["add_section"]["patterns"]):
                match = re.search(r'add\s+(?:a\s+)?(?:section|part)\s+(?:called|named|titled)?\s+["\']?([^"\']+)["\']?', lower_msg)
                section_title = match.group(1) if match else "New Section"
                actions.append({"action": "add_section", "parameters": {"section_title": section_title, "instructions": user_message}, "confidence": 0.7})
            if any(p in lower_msg for p in self.actions["list_sections"]["patterns"]):
                actions.append({"action": "list_sections", "parameters": {}, "confidence": 0.9})
            if any(p in lower_msg for p in self.actions["summarize_doc"]["patterns"]):
                actions.append({"action": "summarize_doc", "parameters": {}, "confidence": 0.8})
            if any(p in lower_msg for p in self.actions["expand_section"]["patterns"]):
                match = re.search(r'(expand|elaborate|extend)\s+(\w+)\s+section', lower_msg)
                section_name = match.group(2) if match else ""
                actions.append({"action": "expand_section", "parameters": {"section_name": section_name, "instructions": user_message}, "confidence": 0.7})
            if any(p in lower_msg for p in self.actions["rewrite_section"]["patterns"]):
                match = re.search(r'(rewrite|rephrase)\s+(\w+)\s+section', lower_msg)
                style_match = re.search(r'(?:in|with|using)\s+(?:a\s+)?(\w+)\s+style', lower_msg)
                section_name = match.group(2) if match else ""
                style = style_match.group(1) if style_match else "formal"
                actions.append({"action": "rewrite_section", "parameters": {"section_name": section_name, "style": style}, "confidence": 0.7})
            
            return actions if actions else [{"action": "list_actions", "parameters": {}, "confidence": 0.6}]

        logger.info(f"Detecting action using thinking model for message: {user_message}")

        system_prompt = "You are an AI assistant helping with document generation and management tasks.\nAvailable actions:\n"
        for a_name, a_info in self.actions.items():
            reqs = []
            for req in a_info.get("requires", []):
                if req.get("type") == "enum":
                    valid = ", ".join([f"'{v}'" for v in req.get("values", [])])
                    reqs.append(f"{req.get('name')} (must be one of: {valid})")
                elif req.get("type") == "boolean":
                    reqs.append(f"{req.get('name')} (true/false)")
                else:
                    reqs.append(f"{req.get('name')} ({req.get('description', '')})")
            system_prompt += f"- {a_name}: {a_info['description']} (requires: {', '.join(reqs)})\n"
        system_prompt += (
            "\nAnalyze the user request and identify ALL actions they want to perform. You can detect multiple actions in a single request.\n"
            "Respond with a JSON array containing objects with these fields:\n"
            "- action: The action name\n- parameters: Relevant parameters (e.g., topic, section_name, etc.)\n- confidence: Your confidence (0.0-1.0)\n"
            "\nExample response for multiple actions:\n"
            "[{'action': 'write_full_doc', 'parameters': {'topic': 'AI ethics', 'template_id': 'research'}, 'confidence': 0.9},"
            "{'action': 'add_section', 'parameters': {'section_title': 'Legal implications', 'instructions': 'Add a section about legal implications'}, 'confidence': 0.8}]"
        )
        user_prompt = f"What document action(s) should I take based on this request: '{user_message}'? Please identify ALL actions in the request."
        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = await self._query_thinking_model(client, system_prompt, user_prompt, extract_json=True, ctx=ctx)
                # Handle both single action (dict) or multiple actions (list)
                if isinstance(results, dict):
                    return [results]
                return results
            except Exception as e:
                logger.error(f"Error detecting action on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                example_json = "[{'action': 'write_full_doc', 'parameters': {'topic': 'example topic', 'template_id': 'basic'}, 'confidence': 0.9}]"
                user_prompt = (f"I need a valid JSON response for this request: '{user_message}'\n"
                               f"Return a JSON array with action objects, like this example:\n{example_json}\n"
                               f"Choose from these actions: {', '.join(self.actions.keys())}")
                await asyncio.sleep(1)
        return [{"action": "list_actions", "parameters": {}, "confidence": 0.5}]

    def _extract_json_from_response(self, content, force_first_element=False):
        # Remove <think> tags
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        json_match = re.search(r'\[\s*{.*?}\s*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        else:
            json_match = re.search(r'{.*}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            # Attempt to convert single quotes to double quotes and try again
            fixed_content = content.replace("'", '"')
            data = json.loads(fixed_content)
        if force_first_element and isinstance(data, list) and data:
            return data[0]
        return data

    async def _stream_response(self, client, base_url, endpoint, payload, ctx, headers, timeout, max_retries=10):
        """
        Common helper for streaming responses from a given base URL.
        Filters out any text within <think>...</think> markers.
        """
        inside_think = False  # tracks if we are currently inside a <think> block
        buffer = ""  # accumulates text across chunks
        for attempt in range(max_retries + 1):
            try:
                async with client.stream("POST", f"{base_url}{endpoint}", json=payload, headers=headers, timeout=timeout) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        if attempt < max_retries:
                            logger.error(f"API error ({response.status_code}), retrying...")
                            await asyncio.sleep(5)
                            continue
                        else:
                            yield {"error": f"API error ({response.status_code}): {error_text.decode('utf-8')}"}
                            return

                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        # Decode and accumulate chunk text
                        chunk_str = chunk.decode('utf-8')
                        buffer += chunk_str

                        # Process complete lines from the buffer
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            # Only process lines that start with the expected prefix and aren't the DONE marker.
                            if line.startswith("data: ") and line.strip() != "data: [DONE]":
                                # Remove the 'data: ' prefix and filter out <think> sections.
                                content = line[6:]
                                filtered_content = ""
                                pos = 0
                                while pos < len(content):
                                    if not inside_think:
                                        # Look for the start of a <think> block.
                                        start_idx = content.find("<think>", pos)
                                        if start_idx == -1:
                                            # No <think> tag: take the rest of the content.
                                            filtered_content += content[pos:]
                                            break
                                        else:
                                            # Append the text before <think> and mark that we're inside.
                                            filtered_content += content[pos:start_idx]
                                            pos = start_idx + len("<think>")
                                            inside_think = True
                                    else:
                                        # Already inside a <think> block: look for its end.
                                        end_idx = content.find("</think>", pos)
                                        if end_idx == -1:
                                            # No closing tag in this chunk: discard the rest.
                                            pos = len(content)
                                            break
                                        else:
                                            pos = end_idx + len("</think>")
                                            inside_think = False
                                if filtered_content.strip():
                                    try:
                                        data = json.loads(filtered_content)
                                        yield data
                                    except json.JSONDecodeError as e:
                                        # logger.error(f"JSON parse error in stream: {e}")
                                        pass
                                else:
                                    logger.debug("Skipped empty filtered content")
                    return
            except httpx.TimeoutException:
                if attempt < max_retries:
                    logger.error("Request timed out, retrying...")
                    wait_time = 2**(attempt+1)
                    ctx["event_emitter"]({
                        "type": "status",
                        "data": {"description": f"Request timed out, retrying in {wait_time} seconds", "done": False}
                    })
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    yield {"error": f"Request timed out after {timeout} seconds"}
                    return
            except Exception as e:
                if attempt < max_retries:
                    logger.error(f"Exception in stream: {str(e)}, retrying...")
                    wait_time = 2**(attempt+1)
                    ctx["event_emitter"]({
                        "type": "status",
                        "data": {"description": f"Exception in stream: {str(e)}, retrying in {wait_time} seconds", "done": False}
                    })
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    yield {"error": str(e)}
                    return

    async def _call_api(self, client, endpoint, payload, ctx, timeout, stream=False):
        """
        Make a call to the regular API_HOST.
        """
        base_url = self.valves.API_HOST
        if stream:
            return self._stream_response(client, base_url, endpoint, payload, ctx, self._get_api_headers(self.valves.API_KEY), timeout)
        else:
            response = await client.post(f"{base_url}{endpoint}", json=payload, headers=self._get_api_headers(self.valves.API_KEY) , timeout=timeout)
            response.raise_for_status()
            return response.json()

    async def _call_thinking_api(self, client, endpoint, payload, ctx, timeout, stream=False):
        """
        Make a call to the THINKING_API_HOST.
        """
        base_url = self.valves.THINKING_API_HOST
        payload["model"] = self.valves.THINKING_MODEL_ID
        logger.info(f"Calling thinking API with payload: {payload}")
        if stream:
            return self._stream_response(client, base_url, endpoint, payload, ctx, self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY), timeout)
        else:
            response = await client.post(f"{base_url}{endpoint}", json=payload, headers=self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY) , timeout=timeout)
            response.raise_for_status()
            return response.json()

    async def generate_section(self, client, section, document, ctx, instructions=None):

        """
        Generate a single section using a streaming API call.
        If use_thinking_for_generation is True, use the thinking model and remove any <think> tags.
        """
        section_type = section.get("sectionType", "free_text")
        system_prompt = self.system_prompts["generic_init"]

        topic = document.get("topic", "")
        language = document.get("language", "english")
        style = document.get("style", "neutral")
        tone = document.get("tone", "neutral")
        length = document.get("length", "medium")

        prompt = section.get("prompt", "").replace("{topic}", topic)

        logger.info(f"Section: {section}, instructions: {instructions}")

        rag_context = ""
        logger.info(f"Generating section: {section['title']}, type: {section_type}, instructions: {instructions}, topic: {topic}, language: {language}, style: {style}, tone: {tone}, length: {length}")
        # Prepare system prompt with RAG context if available
        system_prompt = self.system_prompts["generic_init"]

        if ctx["rag"]["use_rag"] and (section_type == "free_text" or section_type == "populate"):
            rag_system_prompt = self.system_prompts["rag"]
            rag_user_prompt = f"What is the query for getting data from a Vector Database for the following topic: '{topic}' and the following prompt: '{prompt}'?"
            rag_query = await self._query_thinking_model(client, rag_system_prompt, rag_user_prompt, ctx=ctx, extract_json=False)
            rag_query = re.sub(r'<think>.*?</think>', '', rag_query, flags=re.DOTALL)
            logger.info(f"RAG query: {rag_query}")
            rag_context, docs_count = await self._get_rag_context(ctx, topic, rag_query)
            yield {"content_piece": f"RAG query: {rag_query.strip()} ({docs_count} results)\n"}

        if section_type == "fixed":
            yield {"content": section.get("template_text", ""), "status": "complete"}
            return

        elif section_type == "populate":
            template_text = section.get("template_text", "")
            populated_content = template_text
            for field in section.get("populateFields", []):
                field_name = field.get("name", "")
                field_prompt = field.get("prompt", "").format(topic=topic)
                system_prompt += (
                    "You are given a field prompt and a topic. Generate a plain text value for the field without any additional conversation or formatting. "
                    "Return only the final answer and nothing else. the language of the document is {language}."
                    f"This is for the section \"{section['title']}\"."
                )
                if rag_context:
                    system_prompt += f"\n\nThis infomration could help you in writing your response:\n\n{rag_context}\n\n"
                field_value = ""
                try:
                    payload = {
                        "model": self.valves.MODEL_ID,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": field_prompt}
                        ],
                        "temperature": self.valves.TEMPERATURE,
                        "max_tokens": min(200, self.valves.MAX_TOKENS),
                        "stream": False,
                    }
                    # For field generation we always use API_HOST
                    result = await self._call_api(client, "/chat/completions", payload, ctx, self.valves.REQUEST_TIMEOUT, stream=False)
                    field_value = result.get("choices", [])[0].get("message", {}).get("content", "").strip()
                    populated_content = populated_content.replace(f"{{{field_name}}}", field_value)
                    yield {"content_piece": f"Generated value for {field_name}: {field_value}\n"}
                except Exception as e:
                    logger.error(f"Error generating field {field_name}: {e}")
                    populated_content = populated_content.replace(f"{{{field_name}}}", f"[Error generating {field_name}]")
            yield {"content": populated_content, "status": "complete"}
            return

        elif section_type in ["empty", "toc"]:
            if section_type == "toc":
                yield {"content": "[Table of Contents will be generated when displaying the document]", "status": "complete"}
            else:
                yield {"content": "", "status": "complete"}
            return

        document_structure = self._generate_document_outline(section)
        prompt = ""
        logger.info(f"Document structure: {document_structure}, instructions: {instructions}")
        if instructions:
            current_content = document['sections'].get(section['name'], "")
            if current_content:
                prompt = (f"Edit the section \"{section['title']}\" according to these instructions: {instructions}\n\n"
                    f"Current section content:\n{current_content}\n\n"
                    "Provide the complete edited section, only the content, no title or other text.")
            else:
                prompt = (f"Generate a new section \"{section['title']}\" according to these instructions: {instructions}\n\n"
                    "Provide the new section, only the content, no title or other text.")
        else:
            prompt = section["prompt"].format(topic=topic)
        system_prompt += (
            f"{self.system_prompts['no_sections']}"
            f"You are generating content for a document on the topic: '{topic}'.\n"
            f"Document structure:\n{document_structure}\n\n"
            f"Generate or edit, according to the instructions, a single paragraph for the section \"{section['title']}\". "
            f"The language of the document is {language}. "
            f"The style of the document is {style}. "
            f"The tone of the document is {tone}. "
            f"The length of the response should be {length}. "
            f"{self.system_prompts['no_conclusions']}"
        )
        logger.info(f"System prompt: {system_prompt}")
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if rag_context:
            messages.append({"role": "user", "content": f"This infomration could help you in writing your response:\n\n{rag_context}\n\n"})
        
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.valves.MODEL_ID,
            "messages": messages,
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,
        }
        # Determine which host to use based on the flag
        if self.use_thinking_for_generation:
            stream_gen = await self._call_thinking_api(client, "/chat/completions", payload, ctx, self.valves.REQUEST_TIMEOUT, stream=True)
        else:
            stream_gen = await self._call_api(client, "/chat/completions", payload, ctx, self.valves.REQUEST_TIMEOUT, stream=True)

        content_buffer = ""
        async for data in stream_gen:
            if "error" in data:
                logger.error(f"Section generation error: {data['error']}")
                yield {"content": "", "status": "complete"}
                return
            if 'choices' in data and data["choices"]:
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    piece = delta["content"]
                    content_buffer += piece
                    yield {"content_piece": piece}
        # If using thinking model for generation, remove any <think> markers.
        if self.use_thinking_for_generation:
            content_buffer = re.sub(r'<think>.*?</think>', '', content_buffer, flags=re.DOTALL)
        yield {"content": content_buffer, "status": "complete"}

    def _generate_document_outline(self, current_section):
        """Generate a markdown outline based on the document structure."""
        outline = []
        root_sections = None
        if hasattr(self, 'current_document') and self.current_document:
            root_sections = self.current_document.get("template", {}).get("sections", [])
        if not root_sections:
            for _, template in self.templates.items():
                def find_in_sections(sections, section_name):
                    for s in sections:
                        if s["name"] == section_name:
                            return True
                        if s.get("children"):
                            if find_in_sections(s["children"], section_name):
                                return True
                    return False
                if find_in_sections(template["sections"], current_section["name"]):
                    root_sections = template["sections"]
                    break
        if not root_sections:
            outline.append(f"# {current_section['title']}")
            return "\n".join(outline)
        def build_outline(sections, level=1):
            for sec in sections:
                prefix = "#" * level
                outline.append(f"{prefix} {sec['title']}")
                if sec.get("children"):
                    build_outline(sec["children"], level + 1)
        build_outline(root_sections)
        return "\n".join(outline)

    def find_section_in_template(self, sections, section_name):
        for s in sections:
            if s["name"] == section_name:
                return s
            if s.get("children"):
                result = self.find_section_in_template(s["children"], section_name)
                if result:
                    return result
        return None

    async def handle_write_full_doc(self, client, chat_id, user_message, model_id, params, ctx):
        """Handle generating a full document."""
        topic = params.get("topic", user_message).strip()
        title = params.get("title", "").strip()
        language = params.get("language", "english").strip()
        style = params.get("style", "neutral").strip()
        tone = params.get("tone", "neutral").strip()
        length = params.get("length", "medium").strip()
        template_id = params.get("template_id", self.valves.TEMPLATE_NAME)

        if not topic:
            yield {"think": "Error: No topic provided. Please specify a topic for the document."}
            return
        if template_id not in self.templates:
            template_id = self.valves.TEMPLATE_NAME
        template = self.templates[template_id]
        
        if ctx["event_emitter"]:
            await ctx["event_emitter"]({
                "type": "status",
                "data": {"description": f"Generating document on '{topic}'", "done": False}
            })
        yield {"think": f"📒 Generating full document on '{topic}' using template: {template['name']}\n\n"}

        document = {
            "topic": topic,
            "title": title,
            "template": template,
            "language": language,
            "style": style,
            "tone": tone,
            "length": length,
            "model_id": self.valves.MODEL_ID,
            "sections": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }


        async for chunk in self.generate_document(client, document, ctx):
            yield {"think": chunk}
        if hasattr(self, 'current_document') and self.current_document:
            document = self.current_document
            self.current_document = None
        self.documents[chat_id] = document
        self.save_document(chat_id, document)
        yield {"think": "\nCompiling final document...\n"}
        yield {"result": self.format_document(document)}

    async def handle_edit_section(self, client, chat_id, params, ctx=None):
        """Handle editing a specific section."""
        section_name = params.get("section_name", "")
        instructions = params.get("instructions", "Improve this section")
        if chat_id not in self.documents:
            yield {"result": "No document found to edit. Please generate a document first."}
            return
        document = self.documents[chat_id]
        section, _ = self.find_section_by_name(document, section_name)
        if not section or section["name"] not in document["sections"]:
            yield {"result": f"Section '{section_name}' not found in the document.\nAvailable sections:\n" + self.list_document_sections(document)}
            return
        section_title = section["title"]
        section_name = section["name"]
        yield {"think": f"Editing section: {section_title} ({section_name}) with instructions: {instructions}\n\n"}
        if ctx["event_emitter"]:
            await ctx["event_emitter"]({
                "type": "status",
                "data": {"description": f"Editing section: {section_title}", "done": False}
            })
        section_content = ""
        async for chunk in self.generate_section(client, section, document, ctx, instructions):
            if "error" in chunk:
                yield {"result": f"Error editing section: {chunk['error']}\n"}
                break
            elif "content_piece" in chunk:
                section_content += chunk["content_piece"]
                yield {"think": chunk["content_piece"]}
            elif "status" in chunk and chunk["status"] == "complete":
                document["sections"][section_name] = chunk.get("content", section_content)
                document["updated_at"] = time.time()
                self.save_document(chat_id, document)
                yield {"think": f"\nCompleted editing section: {section_title}\n"}
        yield {"result": self.format_document(document, highlight_section=section_name)}

    async def determine_section_position(self, client, document, section_title, position_info, ctx):
        """
        Determine where to place a new section in the document based on user instructions.
        Returns positioning information including parent section and reference section.
        """
        def extract_section_structure(sections, parent=None, level=0):
            result = []
            for sec in sections:
                info = {
                    "name": sec["name"],
                    "title": sec["title"],
                    "level": level,
                    "parent": parent
                }
                result.append(info)
                if sec.get("children"):
                    result.extend(extract_section_structure(sec["children"], parent=sec["name"], level=level+1))
            return result

        doc_structure = extract_section_structure(document["template"]["sections"])
        logger.info(f"Document structure: {doc_structure}")
        system_prompt = (
            "You are an AI assistant helping with document structure organization.\n"
            "Determine where to place a new section in an existing document based on the user's request.\n"
            "Return a JSON object with these fields:\n"
            "- parent_section: The parent section name (or null if top-level)\n"
            "- reference_section: The section name to place the new section before or after\n"
            "- position: Either 'before' or 'after'\n"
            "- is_nested_sub_section: Whether the new section is nested under the reference section\n"
            "- explanation: A brief explanation for the chosen position\n"
            "Example:\n"
            '{ "parent_section": null, "reference_section": "introduction", "position": "after", "is_nested_sub_section": false, "explanation": "User requested to add after introduction." }'
        )
        user_prompt = (
            f"Current document structure:\n{json.dumps(doc_structure, indent=2)}\n\n"
            f"New section title: \"{section_title}\"\nUser's positioning request: \"{position_info}\"\n"
            "Return the positioning as a JSON object."
        )
        try:
            position_data = await self._query_thinking_model(client, system_prompt, user_prompt, ctx=ctx, extract_json=True, force_first_element=True)
            for field in ["reference_section", "position"]:
                if field not in position_data:
                    position_data[field] = "end" if field == "reference_section" else "after"
            return position_data
        except Exception as e:
            logger.error(f"Error determining section position: {e}")
            return {"parent_section": None, "reference_section": "end", "position": "after", "explanation": f"Default due to error: {e}"}

    def insert_section_into_document(self, document, section_metadata, position_data): 
        """
        Insert a new section into the document template at the specified position.
        """
        try:
            logger.info(f"Inserting section: {section_metadata['name']} ({section_metadata['title']})")
            logger.info(f"Position data: {position_data}")
            reference_section = position_data.get("reference_section")
            position = position_data.get("position", "after")
            parent_section = position_data.get("parent_section")
            is_nested_sub_section = position_data.get("is_nested_sub_section", False)

            # If the insertion is nested and a parent section is provided:
            if is_nested_sub_section and parent_section:
                # Helper function to locate a section by name recursively.
                def find_section(sections, target):
                    for sec in sections:
                        if sec.get("name") == target:
                            return sec
                        if sec.get("children"):
                            found = find_section(sec["children"], target)
                            if found:
                                return found
                    return None

                parent_node = find_section(document["template"]["sections"], parent_section)
                if parent_node is None:
                    logger.warning(f"Parent section {parent_section} not found. Inserting at root.")
                    document["template"]["sections"].append(section_metadata)
                    return True

                # If a reference_section is provided, try to insert relative to it within parent's children.
                if reference_section:
                    children = parent_node.setdefault("children", [])
                    # Find the index of the reference_section among the parent's children.
                    index = next((i for i, child in enumerate(children) if child.get("name") == reference_section), None)
                    if index is not None:
                        if position == "before":
                            children.insert(index, section_metadata)
                        else:  # position == "after"
                            children.insert(index + 1, section_metadata)
                        logger.info(f"Inserted nested section under '{parent_section}' relative to '{reference_section}'.")
                        return True
                    else:
                        # If reference not found among parent's children, simply append.
                        children.append(section_metadata)
                        logger.info(f"Reference section '{reference_section}' not found in parent '{parent_section}'. Appended section as nested subsection.")
                        return True
                else:
                    # No reference_section provided: append to the parent's children.
                    parent_node.setdefault("children", []).append(section_metadata)
                    logger.info(f"Inserted nested section under '{parent_section}' at the end.")
                    return True

            # For non-nested insertion or when parent_section is not provided:
            if reference_section == "end" or not reference_section:
                logger.info("Adding section to the end of document")
                document["template"]["sections"].append(section_metadata)
                return True

            # Otherwise, search the entire document structure for the reference_section (non-nested insertion).
            def find_and_insert(sections, parent_name=None):
                for i, sec in enumerate(sections):
                    if sec.get("name") == reference_section:
                        logger.info(f"Found reference section: {sec['name']} at level {parent_name or 'root'}")
                        if position == "before":
                            sections.insert(i, section_metadata)
                        else:
                            sections.insert(i + 1, section_metadata)
                        return True
                    if sec.get("children"):
                        if find_and_insert(sec["children"], sec.get("name")):
                            return True
                return False

            success = find_and_insert(document["template"]["sections"])
            if not success:
                logger.warning(f"Reference section '{reference_section}' not found. Adding at end.")
                document["template"]["sections"].append(section_metadata)
            return True

        except Exception as e:
            logger.error(f"Error inserting section: {e}")
            try:
                document["template"]["sections"].append(section_metadata)
                return True
            except Exception:
                return False

    async def handle_add_section(self, client, chat_id, user_message, params, ctx):
        """Handle adding a new section to the document."""

        section_title = params.get("section_title", "New Section")
        section= {
            "name": re.sub(r'[^a-z0-9]', '', section_title.lower()),
            "title": section_title,
            "sectionType": "free_text",
            "prompt": params.get("instructions", "Add content for this section"),
            "children": []
        }
        section_name = section["name"]

        yield {"think": f"Adding new section: {section_title} with instructions: {section['prompt']}\n\n"}
        yield {"think": f"Analyzing where to place the section based on: {user_message}\n"}
        try:
            position_data = await self.determine_section_position(client, self.documents[chat_id], section_title, user_message, ctx)
            yield {"think": f"Position determined: {json.dumps(position_data, indent=2)}\n\n"}
        except Exception as e:
            logger.error(f"Error in position determination: {e}")
            position_data = {"parent_section": None, "reference_section": "end", "position": "after", "explanation": "Default due to error"}
            yield {"think": f"Error determining position: {e}. Adding to the end of document.\n\n"}
        if ctx["event_emitter"]:
            await ctx["event_emitter"]({
                "type": "status",
                "data": {"description": f"Adding section: {section_title}", "done": False}
            })
        section_content = ""
        document = self.documents[chat_id]

        async for chunk in self.generate_section(client, section, document, ctx, instructions=section['prompt']):
            if "error" in chunk:
                yield {"think": f"Error adding section: {chunk['error']}\n"}
                break
            elif "content_piece" in chunk:
                section_content += chunk["content_piece"]
                yield {"think": chunk['content_piece']}
            elif "status" in chunk and chunk["status"] == "complete":
                section_content = chunk.get("content", section_content)
                document["sections"][section_name] = section_content
                success = self.insert_section_into_document(document, section, position_data)
                if success:
                    document["updated_at"] = time.time()
                    self.save_document(chat_id, document)
                    yield {"think": f"\nCompleted adding section: {section_title}\nSection placed {position_data.get('position', 'after')} {position_data.get('reference_section', 'the end')}\n"}
                else:
                    yield {"think": "\nError positioning the section. Added at the end instead.\n"}
                    document["template"]["sections"].append(section)
                    document["updated_at"] = time.time()
                    self.save_document(chat_id, document)
        yield {"result": self.format_document(document, highlight_section=section_name)}

    async def handle_list_sections(self, chat_id):
        """Handle listing all sections in the document."""
        if chat_id not in self.documents:
            yield {"result": "No document found. Please generate a document first."}
            return
        document = self.documents[chat_id]
        yield {"result": f"Listing sections for document on topic: {document['topic']}\n\n"}
        yield {"result": self.list_document_sections(document)}

    async def handle_summarize_doc(self, client, chat_id, params):
        """Handle summarizing the existing document."""
        yield {"think": "Summarize document action not fully implemented yet."}
        yield {"result": "# Document Summary\n\nThis feature is coming soon!"}

    async def handle_expand_section(self, client, chat_id, params):
        """Handle expanding a specific section with more details."""
        yield {"think": "Expand section action not fully implemented yet."}
        yield {"result": "# Section Expansion\n\nThis feature is coming soon!"}

    async def handle_rewrite_section(self, client, chat_id, params):
        """Handle rewriting a section in a different style."""
        yield {"think": "Rewrite section action not fully implemented yet."}
        yield {"result": "# Section Rewriting\n\nThis feature is coming soon!"}

    def format_document(self, document, highlight_section=None, include_metadata=False):
        """
        Format the document as markdown for display with support for nested sections.
        """
        output = []
        if "title" in document:
            output.append(f"# {document['title']}\n")
        else:
            output.append("# Generated Document\n")
        toc_entries = []

        if include_metadata:
            output.append("\n")
            for meta in self.document_metadata:
                output.append(f"{meta['name']}: {document.get(meta['name'], 'Not provided')}\n")
            output.append("\n")

        def collect_toc_entries(sections, indent=0):
            entries = []
            for sec in sections:
                if sec.get("sectionType") == "toc":
                    continue
                entries.append({"title": sec["title"], "name": sec["name"], "indent": indent})
                if sec.get("children"):
                    entries.extend(collect_toc_entries(sec["children"], indent + 1))
            return entries

        toc_entries = collect_toc_entries(document["template"]["sections"])

        def format_section(sec, level=2):
            sec_key = sec["name"]
            heading = "#" * level
            sec_type = sec.get("sectionType", "free_text")

            if sec_key in document["sections"] or sec_type in ["empty", "toc"] or (sec.get("children") and len(sec["children"]) > 0):
                if sec_type == "toc":
                    output.append(f"{heading} {sec['title']}\n")
                    for entry in toc_entries:
                        output.append(f"[{'-' * entry['indent']}{entry['title']}](#{entry['name']})\n")
                    output.append("\n")
                else:
                    label = ""
                    output.append(f"{heading} {sec['title']} {label}\u007b#{sec['name']}\u007d\n\n")
                    if sec_type != "empty" and sec_key in document["sections"]:
                        output.append(f"{document['sections'][sec_key]}\n\n")
            if sec.get("children"):
                for child in sec["children"]:
                    format_section(child, level + 1)
            else:
                logger.error(f"Section {sec['title']} not found in document, section type: {sec_type}, section key: {sec_key}, sections are: {', '.join(document['sections'].keys())}")

        for sec in document["template"]["sections"]:
            format_section(sec)
        if include_metadata:
            output.append("---\n")
            if "created_at" in document:
                output.append(f"Document created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['created_at']))}\n")
            if "updated_at" in document:
                output.append(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['updated_at']))}\n")
            if "model_id" in document:
                output.append(f"Generated using model: {document['model_id']}\n")
            if "template" in document and "name" in document["template"]:
                output.append(f"Template: {document['template']['name']}\n")
        return "\n".join(output)

    def _get_document_path(self, chat_id):
        """Get the file path for a document based on chat ID."""
        os.makedirs("/app/backend/data/doc_generator_pipe", exist_ok=True)
        return f"/app/backend/data/doc_generator_pipe/{chat_id}.json"

    def save_document(self, chat_id, document):
        """Save a document to file storage if enabled."""
        if self.valves.DOCUMENT_STORAGE != "file":
            return
        try:
            file_path = self._get_document_path(chat_id)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            logger.info(f"Document saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving document: {e}")

    def load_document(self, chat_id):
        """Load a document from file storage if enabled."""
        if self.valves.DOCUMENT_STORAGE != "file":
            return None
        try:
            file_path = self._get_document_path(chat_id)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    document = json.load(f)
                logger.info(f"Document loaded from {file_path}")
                return document
            else:
                logger.info(f"No document found at {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading document: {e}")
            return None

    async def generate_document(self, client, document, ctx):
        """Generate a complete document with nested sections."""

        async def generate_section_recursive(sec, parent_path=""):
            sec_name = sec["title"]
            sec_key = sec["name"]
            yield f"Generating section: {sec_name}...\n"
            section_content = ""
            has_error = False
            async for chunk in self.generate_section(client, sec, document, ctx):
                if "error" in chunk:
                    yield f"Failed to generate section '{sec_name}'. Moving to next section...\n"
                    has_error = True
                    break
                elif "content_piece" in chunk:
                    section_content += chunk["content_piece"]
                    yield chunk["content_piece"]
                elif "status" in chunk and chunk["status"] == "complete":
                    document["sections"][sec_key] = chunk.get("content", section_content)
                    yield f"\nCompleted section: {sec_name}\n"
            if not has_error and sec_key not in document["sections"] and section_content:
                document["sections"][sec_key] = section_content
                yield f"\nCompleted section: {sec_name}\n"
            if sec.get("children"):
                for child in sec["children"]:
                    async for piece in generate_section_recursive(child, f"{parent_path}/{sec_key}"):
                        yield piece

        for sec in document["template"]["sections"]:
            async for piece in generate_section_recursive(sec):
                yield piece
        self.current_document = document

    def find_section_by_name(self, document, section_name):
        """Find a section by name or title in a nested document structure."""
        section_name_lower = section_name.lower()
        logger.info(f"Searching for section: {section_name} in document structure")
        def search(sections, parent_path=""):
            for sec in sections:
                current_name = sec["name"]
                current_path = f"{parent_path}/{current_name}" if parent_path else current_name
                if current_name.lower() == section_name_lower or sec["title"].lower() == section_name_lower:
                    return sec, current_path
                if sec.get("children"):
                    res = search(sec["children"], current_path)
                    if res[0] is not None:
                        return res
            return None, None
        return search(document["template"]["sections"])

    def list_document_sections(self, document):
        """Generate a markdown listing of all sections in the document."""
        output = [f"# Document Sections for: {document.get('title', 'Untitled')}\n"]
        def format_listing(sections, indent=0):
            logger.info(f"Sections: {sections}")
            for sec in sections:
                sec_name = sec["name"]
                sec_title = sec["title"]
                prefix = "  " * indent + "- "
                if sec_name in document["sections"]:
                    word_count = len(document["sections"][sec_name].split())
                    output.append(f"{prefix}**{sec_title}** ({word_count} words) `{sec_name}`\n")
                else:
                    output.append(f"{prefix}{sec_title} (Empty))\n")
                if sec.get("children"):
                    format_listing(sec["children"], indent + 1)
        format_listing(document["template"]["sections"])
        output.append("---\n")
        output.append(f"\n**Title:** {document.get('title', 'Unspecified')}\n")
        output.append(f"**Template:** {document['template']['name']}\n")
        if "created_at" in document:
            output.append(f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['created_at']))}\n")
        if "updated_at" in document:
            output.append(f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['updated_at']))}\n")
        return "\n".join(output)

    def load_git_templates(self):
        """Load additional templates from specified Git repositories."""
        logger.info(f"Loading templates from Git repositories: {self.valves.GIT_TEMPLATE_REPOS}")
        if not self.valves.GIT_TEMPLATE_REPOS.strip():
            logger.info("No Git repositories specified for templates")
            return
        repo_urls = [url.strip() for url in self.valves.GIT_TEMPLATE_REPOS.split(",") if url.strip()]
        logger.info(f"Found {len(repo_urls)} repositories to process")
        for repo_url in repo_urls:
            try:
                logger.info(f"Processing Git repository: {repo_url}")
                repo_name = self._get_repo_name(repo_url)
                repo_dir = f"/app/backend/data/doc_generator_pipe/git/{repo_name}"
                self._clone_or_update_repo(repo_url, repo_dir)
                self._copy_template_files(repo_dir)
            except Exception as e:
                logger.error(f"Error processing Git repository {repo_url}: {e}")

    def _get_repo_name(self, repo_url):
        """Extract repository name from URL."""
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        repo_name = re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name)
        return repo_name

    def _clone_or_update_repo(self, repo_url, repo_dir):
        """Clone a Git repository or update it if it exists."""
        try:
            if os.path.exists(repo_dir):
                logger.info(f"Updating repository: {repo_dir}")
                result = subprocess.run(["git", "-C", repo_dir, "pull"], check=True, capture_output=True, text=True)
                logger.info(f"Git pull: {result.stdout}")
            else:
                logger.info(f"Cloning repository: {repo_url} to {repo_dir}")
                result = subprocess.run(["git", "clone", repo_url, repo_dir], check=True, capture_output=True, text=True)
                logger.info(f"Git clone: {result.stdout}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error in Git operation: {e}")
            return False

    def _copy_template_files(self, repo_dir):
        """Copy template JSON files from the repository to the templates directory."""
        templates_path = os.path.join(repo_dir, "templates")
        if not os.path.exists(templates_path):
            logger.warning(f"No templates directory found in {repo_dir}")
            return
        template_files = glob.glob(os.path.join(templates_path, "*.json"))
        logger.info(f"Found {len(template_files)} template files in {templates_path}")
        for template_file in template_files:
            try:
                template_filename = os.path.basename(template_file)
                timestamp = int(time.time())
                # timestamped_filename = f"{os.path.splitext(template_filename)[0]}_{timestamp}.json"
                timestamped_filename = f"{os.path.splitext(template_filename)[0]}.json"
                destination = f"/app/backend/data/doc_generator_pipe/templates/{timestamped_filename}"
                shutil.copy2(template_file, destination)
                logger.info(f"Copied template: {template_file} -> {destination}")
            except Exception as e:
                logger.error(f"Error copying template file {template_file}: {e}")

    def _load_local_templates(self):
        """Load templates from the local templates directory."""
        template_files = glob.glob("/app/backend/data/doc_generator_pipe/templates/*.json")
        logger.info(f"Loading {len(template_files)} templates from local directory")
        for template_file in template_files:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    template_data = json.load(f)
                if not isinstance(template_data, dict) or "name" not in template_data or "sections" not in template_data:
                    logger.warning(f"Invalid template format in {template_file}. Skipping.")
                    continue
                template_id = os.path.splitext(os.path.basename(template_file))[0]
                self.templates[template_id] = template_data
                logger.info(f"Added template: {template_id} ({template_data.get('name')})")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in template file {template_file}")
            except Exception as e:
                logger.error(f"Error loading template from {template_file}: {e}")

    async def handle_generate_tags(self, client, message, ctx):
        """Handle generating tags for the chat content."""
        logger.info("Generating tags for chat content")
        system_prompt = (
            "You are an expert at categorizing content. Generate appropriate tags that summarize the main themes discussed. "
            "Return only a JSON object with a 'tags' field."
        )
        try:
            content = await self._query_thinking_model(client, system_prompt, message, ctx=ctx, extract_json=True)
            logger.info(f"Generated tags: {content}")
            return content
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return '{"tags": ["General"]}'

    async def handle_list_actions(self):
        """Handle listing all available actions."""
        yield {"think": "Listing all available actions for document generation...\n"}
        output = ["# Available Document Generator Actions\n\n", "The following actions are available:\n\n"]
        for action_name, action_info in self.actions.items():
            output.append(f"## {action_name}\n")
            output.append(f"{action_info['description']}\n\n")
            if action_info.get("requires"):
                output.append("**Required parameters:**\n")
                for req in action_info["requires"]:
                    output.append(f"- {req.get('name', '')}: {req.get('description', '')}\n")
                output.append("\n")
            if action_name == "write_full_doc":
                output.append("**Example:** Write a document about artificial intelligence using the research template.\n\n")
            elif action_name == "edit_section":
                output.append("**Example:** Edit the introduction section to include recent developments.\n\n")
            elif action_name == "add_section":
                output.append("**Example:** Add a section called 'Future Trends' with predictions for the next decade.\n\n")
            elif action_name == "list_sections":
                output.append("**Example:** Show all sections in my document.\n\n")
            elif action_name == "summarize_doc":
                output.append("**Example:** Summarize my current document.\n\n")
            elif action_name == "get_all_templates":
                output.append("**Example:** Show all available document templates.\n\n")
            elif action_name == "generate_template":
                output.append("**Example:** Generate a template from this markdown document:\n\n")
                output.append("```markdown\n# Introduction\nThis is an introduction.\n\n# Terms\nFixed content example.\n```\n")
                output.append("**Section Types:**\n")
                output.append("- Regular: dynamically generated content\n")
                output.append("- Fixed: unchanged content\n")
                output.append("- Populate: content with placeholders\n")
                output.append("- TOC: table of contents\n")
                output.append("- Empty: no content\n\n")
            else:
                output.append("\n")
        yield {"result": "\n".join(output)}

    async def handle_get_all_templates(self):
        """Handle displaying all available templates."""
        yield {"think": "Retrieving all available document templates...\n"}
        output = ["# Available Document Templates\n\n", "The following templates are available:\n\n"]
        for template_id, template in self.templates.items():
            output.append(f"## {template['name']} ({template_id})\n")
            if "description" in template:
                output.append(f"{template['description']}\n\n")
            output.append("**Sections:**\n")
            def format_sections(sections, indent=0):
                section_list = []
                for sec in sections:
                    indent_str = "  " * indent
                    if sec.get("sectionType") == "fixed":
                        section_list.append(f"{indent_str}- {sec['title']} (Fixed Section)")
                    elif sec.get("sectionType") == "populate":
                        fields = [f['name'] for f in sec.get('populateFields', [])]
                        fields_str = ", ".join(fields) if fields else "none"
                        section_list.append(f"{indent_str}- {sec['title']} (Template with fields: {fields_str})")
                    else:
                        section_list.append(f"{indent_str}- {sec['title']}")
                    if sec.get("children"):
                        section_list.extend(format_sections(sec["children"], indent + 1))
                return section_list
            section_list = format_sections(template["sections"])
            output.extend([f"{s}\n" for s in section_list])
            output.append("\n")
        output.append("To use a template, request: Write a document about [topic] using the [template_id] template.\n")
        output.append("\n**Section Types:**\n")
        output.append("- Regular: dynamically generated content\n")
        output.append("- Fixed: predefined content\n")
        output.append("- Populate: placeholders to be populated\n")
        yield {"result": "\n".join(output)}

    async def generate_template_from_markdown(self, client, user_message):
        """
        Generate a template from a markdown document provided by the user.
        """
        yield {"think": "Analyzing markdown document to generate a template...\n"}
        markdown_lines = user_message.strip().split('\n')
        if not any(line.startswith('#') for line in markdown_lines):
            yield {"think": "Error: No markdown headings found. Please provide a document with headings starting with #.\n"}
            yield {"result": "# Template Generation Failed\n\nNo headings found in the document."}
            return
        section_tree = []
        current_sections = {0: None}
        section_content = {}
        current_section_type = "free_text"
        current_section_populate_fields = []
        for line in markdown_lines:
            if line.startswith('#'):
                logger.info(f"Processing line: {line}")
                match = re.match(r'#+\s*(\{\{.*?\}\})\s*(.*)', line)
                if match:
                    tag = match.group(1)
                    title = match.group(2)
                    if tag == "{{fixed}}":
                        current_section_type = "fixed"
                        current_section_populate_fields = []
                    elif tag == "{{empty}}":
                        current_section_type = "empty"
                        current_section_populate_fields = []
                    elif tag == "{{toc}}":
                        current_section_type = "toc"
                        current_section_populate_fields = []
                    elif tag.startswith("{{populate:"):
                        current_section_type = "populate"
                        try:
                            fields_json = tag[11:-2].strip()
                            current_section_populate_fields = json.loads(fields_json)
                        except Exception as e:
                            logger.error(f"Error parsing populate fields: {e}")
                            current_section_populate_fields = []
                            yield {"think": f"Warning: Error parsing populate fields in '{title}'.\n"}
                    else:
                        current_section_type = "free_text"
                        current_section_populate_fields = []
                    line = f"# {title}"
                else:
                    current_section_type = "free_text"
                    current_section_populate_fields = []
                level = len(re.match(r'(#+)', line).group(1))
                title = line[level:].strip()
                section_name = re.sub(r'[^a-z0-9_]', '', title.lower().replace(' ', '_'))
                if not section_name:
                    section_name = f"section_{len(section_content)}"
                base_name = section_name
                counter = 1
                while section_name in section_content:
                    section_name = f"{base_name}_{counter}"
                    counter += 1
                new_section = {
                    'name': section_name,
                    'title': title,
                    'prompt': '',
                    'sectionType': current_section_type,
                    'children': []
                }
                if current_section_type == "populate":
                    new_section['populateFields'] = current_section_populate_fields
                section_content[section_name] = []
                if level == 1:
                    section_tree.append(new_section)
                    current_sections = {1: new_section}
                else:
                    parent_level = level - 1
                    while parent_level > 0 and parent_level not in current_sections:
                        parent_level -= 1
                    if parent_level > 0 and parent_level in current_sections:
                        current_sections[parent_level].setdefault('children', []).append(new_section)
                    else:
                        section_tree.append(new_section)
                current_sections[level] = new_section
                for k in list(current_sections.keys()):
                    if k > level:
                        del current_sections[k]
            elif current_sections.get(1) is not None:
                max_level = max(current_sections.keys())
                current_section = current_sections[max_level]
                section_content[current_section['name']].append(line)
        all_sections = []
        def collect_sections(sections):
            for sec in sections:
                all_sections.append(sec)
                if sec.get('children'):
                    collect_sections(sec['children'])
        collect_sections(section_tree)
        yield {"think": f"Extracted {len(all_sections)} sections from the document.\n"}
        for sec in all_sections:
            sec_name = sec['name']
            sec_title = sec['title']
            content = "\n".join(section_content.get(sec_name, []))
            sec_type = sec.get('sectionType', 'generate')
            if sec_type == "fixed":
                sec['template_text'] = content
                sec['prompt'] = f"Fixed section: {sec_title}"
                yield {"think": f"Configured fixed section: '{sec_title}'\n"}
            elif sec_type == "populate":
                sec['template_text'] = content
                sec['prompt'] = f"Populate template for {sec_title} with field values"
                yield {"think": f"Configured populate section: '{sec_title}' with {len(sec.get('populateFields', []))} fields\n"}
            else:
                if not content.strip() and sec.get('children'):
                    sec['prompt'] = f"Provide information about {{topic}} related to {sec_title}."
                    continue
                system_prompt = (
                    "You are an AI assistant creating document templates.\n"
                    "Based on the section title and sample content, create a concise prompt (25-40 words) that includes a {topic} placeholder."
                )
                user_prompt = f"Section title: {sec_title}\n\nSample content:\n{content}\n\nCreate a brief prompt for this section."
                payload = {
                    "model": self.valves.THINKING_MODEL_ID,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.3
                }
                try:
                    response = await client.post(
                        f"{self.valves.THINKING_API_HOST}/chat/completions",
                        headers=self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY),
                        json=payload,
                        timeout=30
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    prompt_text = response_data.get("choices", [])[0].get("message", {}).get("content", "").strip()
                    prompt_text = prompt_text.strip('`"')
                    if "{topic}" not in prompt_text:
                        prompt_text = f"Discuss {sec_title.lower()} for {{topic}}."
                    if not prompt_text.endswith('.'):
                        prompt_text += '.'
                    sec['prompt'] = prompt_text
                    yield {"think": f"Generated prompt for '{sec_title}': {prompt_text}\n"}
                except Exception as e:
                    logger.error(f"Error generating prompt for section {sec_name}: {e}")
                    sec['prompt'] = f"Explain {sec_title.lower()} for {{topic}}."
                    yield {"think": f"Error generating prompt for '{sec_title}': {e}\n"}
        yield {"think": "Generating template metadata...\n"}
        template_name = "Custom Template"
        template_description = "Generated from user document"
        try:
            system_prompt = (
                "You are an AI assistant creating document templates.\n"
                "Based on the document structure, suggest a concise name and brief description for this template. "
                "Return a JSON object with 'name' and 'description' fields."
            )
            section_titles = []
            def extract_titles(sections, indent=0):
                for sec in sections:
                    section_titles.append("  " * indent + sec['title'])
                    if sec.get('children'):
                        extract_titles(sec['children'], indent + 1)
            extract_titles(section_tree)
            user_prompt = f"Document section titles:\n{json.dumps(section_titles, indent=2)}\n\nCreate a name and description for this template. Return only JSON."
            payload = {
                "model": self.valves.THINKING_MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.3
            }
            response = await client.post(
                f"{self.valves.THINKING_API_HOST}/chat/completions",
                headers=self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY),
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            response_data = response.json()
            metadata_str = response_data.get("choices", [])[0].get("message", {}).get("content", "")
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', metadata_str)
            if json_match:
                metadata_str = json_match.group(1)
            else:
                json_match = re.search(r'(\{[\s\S]*\})', metadata_str)
                if json_match:
                    metadata_str = json_match.group(1)
            metadata = json.loads(metadata_str)
            if isinstance(metadata, list) and metadata:
                metadata = metadata[0]
            template_name = metadata.get("name", template_name)
            template_description = metadata.get("description", template_description)
            yield {"think": f"Generated template name: {template_name}\n"}
            yield {"think": f"Generated description: {template_description}\n"}
        except Exception as e:
            logger.error(f"Error generating template metadata: {e}")
            yield {"think": f"Error generating template metadata: {e}\n"}
        template_id = re.sub(r'[^a-z0-9_]', '', template_name.lower().replace(' ', '_'))
        base_id = template_id
        counter = 1
        while template_id in self.templates:
            template_id = f"{base_id}_{counter}"
            counter += 1
        template = {"name": template_name, "description": template_description, "sections": section_tree}
        self.templates[template_id] = template
        try:
            os.makedirs("/app/backend/data/doc_generator_pipe/templates", exist_ok=True)
            template_path = os.path.join("/app/backend/data/doc_generator_pipe/templates", f"{template_id}.json")
            with open(template_path, 'w') as f:
                json.dump(template, f, indent=2)
            yield {"think": f"Template saved to {template_path}\n"}
        except Exception as e:
            logger.error(f"Error saving template: {e}")
            yield {"think": f"Error saving template: {e}\n"}

        output = [f"# Template Generated: {template_name}\n\n", f"{template_description}\n\n", f"**Template ID:** `{template_id}`\n\n", "**Sections:**\n"]
        def format_template_sections(sections, indent=0):
            section_list = []
            for sec in sections:
                indent_str = "  " * indent
                section_list.append(f"{indent_str}- **{sec['title']}**\n{indent_str}  Prompt: _{sec['prompt']}_")
                if sec.get("children"):
                    section_list.extend(format_template_sections(sec["children"], indent + 1))
            return section_list
        section_list = format_template_sections(template["sections"])
        output.extend([f"{s}\n" for s in section_list])
        output.append("\n")
        output.append(f"To use this template, request: Write a document about [topic] using the {template_id} template.\n")
        yield {"result": "\n".join(output)}

    def _get_api_headers(self, api_key):
        """Return API headers with the specified API key."""
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openwebui.com/",
            "X-Title": "Open WebUI",
        }

    async def handle_print_document(self, chat_id):
        """Handle printing the current document."""
        if chat_id not in self.documents:
            yield {"think": "No document found. Please generate a document first."}
            yield {"result": "# No Document Found\n\nPlease create a document using the 'write_full_doc' action."}
            return
        document = self.documents[chat_id]
        yield {"think": f"Retrieving document on topic: {document.get('topic', 'Untitled')}\n"}
        full_doc = "Full Document: {#fulldoc}\n\n" + f"{self.format_document(document, include_metadata=False)}\n"
        yield {"result": full_doc}

    async def handle_delete_template(self, params):
        """Handle deleting a template."""
        template_id = params.get("template_id", "")
        yield {"think": f"Attempting to delete template: {template_id}...\n"}
        if template_id not in self.templates:
            yield {"think": f"Error: Template '{template_id}' not found.\n"}
            yield {"result": "# Template Not Found\n\nNo template with ID '{template_id}' was found."}
            return
        builtin_templates = ["basic", "research", "advanced", "custom"]
        if template_id in builtin_templates:
            yield {"think": f"Error: Cannot delete built-in template '{template_id}'.\n"}
            yield {"result": "# Cannot Delete Built-in Template\n\nTemplate '{template_id}' is built-in and cannot be deleted."}
            return
        template_name = self.templates[template_id].get("name", template_id)
        file_deleted = False
        template_files = glob.glob(f"/app/backend/data/doc_generator_pipe/templates/{template_id}.json")
        for file_path in template_files:
            try:
                os.remove(file_path)
                file_deleted = True
                yield {"think": f"Deleted template file: {file_path}\n"}
            except Exception as e:
                yield {"think": f"Warning: Failed to delete template file {file_path}: {e}\n"}
        try:
            del self.templates[template_id]
            yield {"think": f"Removed template '{template_id}' from memory.\n"}
            yield {"result": f"# Template Deleted\n\nTemplate '{template_name}' (ID: {template_id}) has been successfully deleted."}
            if file_deleted:
                yield {"think": "\n\nTemplate file was also removed."}
        except Exception as e:
            yield {"think": f"Error removing template from memory: {e}\n"}
            yield {"result": f"# Template Deletion Error\n\nAn error occurred while deleting template '{template_name}' (ID: {template_id}): {e}"}

    async def handle_get_template_json(self, params):
        """Handle getting a template's JSON structure."""
        template_id = params.get("template_id", "")
        yield {"think": f"Retrieving JSON for template: {template_id}...\n"}
        if template_id not in self.templates:
            yield {"think": f"Error: Template '{template_id}' not found.\n"}
            yield {"result": "# Template Not Found\n\nNo template with ID '{template_id}' was found."}
            return
        template = self.templates[template_id]
        template_json = json.dumps(template, indent=2)
        yield {"think": f"Successfully retrieved template JSON for '{template_id}'.\n"}
        yield {"think": f"# Template: {template['name']} (ID: {template_id})\n\n"}
        if "description" in template:
            yield {"think": f"{template['description']}\n\n"}
        yield {"result": "## Template JSON Structure\n\n"}
        yield {"result": f"```json\n{template_json}\n```\n\n"}
        yield {"result": "You can use this JSON to inspect or create a new template."}

    async def handle_update_template(self, params):
        """Handle updating a template with provided JSON."""
        template_id = params.get("template_id", "")
        template_json_str = params.get("template_json", "")
        yield {"think": f"Attempting to update template: {template_id}...\n"}
        if template_id not in self.templates:
            yield {"think": f"Template '{template_id}' does not exist. This will create a new template.\n"}
        try:
            if "```json" in template_json_str:
                json_match = re.search(r'```json\s*(.*?)\s*```', template_json_str, re.DOTALL)
                if json_match:
                    template_json_str = json_match.group(1)
            elif "```" in template_json_str:
                json_match = re.search(r'```\s*(.*?)\s*```', template_json_str, re.DOTALL)
                if json_match:
                    template_json_str = json_match.group(1)
            template_data = json.loads(template_json_str)
            if not isinstance(template_data, dict):
                raise ValueError("Template must be a JSON object")
            if "name" not in template_data:
                raise ValueError("Template must have a 'name' field")
            if "sections" not in template_data:
                raise ValueError("Template must have a 'sections' field")
            if not isinstance(template_data["sections"], list):
                raise ValueError("Template 'sections' must be an array")
            for sec in template_data["sections"]:
                if "name" not in sec or "title" not in sec:
                    raise ValueError("Each section must have 'name' and 'title' fields")
            builtin_templates = ["basic", "research", "advanced", "custom", "test_advanced_fields"]
            if template_id in builtin_templates:
                yield {"think": f"Warning: Updating built-in template '{template_id}'. It will be updated in memory only.\n"}
            self.templates[template_id] = template_data
            yield {"think": f"Updated template '{template_id}' in memory.\n"}
            if template_id not in builtin_templates:
                try:
                    file_path = f"/app/backend/data/doc_generator_pipe/templates/{template_id}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(template_data, f, ensure_ascii=False, indent=2)
                    yield {"think": f"Saved template to file: {file_path}\n"}
                except Exception as e:
                    yield {"think": f"Warning: Failed to save template to file: {e}\n"}
            yield {"think": f"# Template Updated Successfully\n\nThe template '{template_data['name']}' (ID: {template_id}) has been updated.\n\n"}
            yield {"result": f"## Template Summary\n\n**Name:** {template_data['name']}\n"}
            if "description" in template_data:
                yield {"result": f"**Description:** {template_data['description']}\n"}
            yield {"result": f"**Number of Sections:** {len(template_data['sections'])}\n\n"}
            yield {"result": "### Sections\n\n"}
            for sec in template_data["sections"]:
                yield {"result": f"- {sec['title']}\n"}
        except json.JSONDecodeError as e:
            yield {"think": f"Error: Invalid JSON format: {e}\n"}
            yield {"result": "# Template Update Failed\n\nInvalid JSON: {e}\n\nPlease correct and try again."}
        except ValueError as e:
            yield {"think": f"Error: {e}\n"}
            yield {"result": "# Template Update Failed\n\n{e}\n\nPlease correct and try again."}
        except Exception as e:
            yield {"think": f"Error updating template: {e}\n"}
            yield {"result": "# Template Update Failed\n\nAn unexpected error occurred: {e}"}

    async def _query_thinking_model(self, client, system_prompt, user_prompt, extract_json=False, temperature=0.3, force_first_element=False, ctx=None, **kwargs):
        """
        Query the thinking model and clean the response.
        """
        start_time = asyncio.get_event_loop().time()
        if not ctx:
            logger.error("No context provided for thinking model query")
            raise ValueError("No context provided for thinking model query")
        try:
            payload = {
                "model": self.valves.THINKING_MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ] + kwargs.get("messages", []),
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", self.valves.MAX_THINKING_TOKENS)
            }
            logger.info(f"Payload Messages: {payload['messages']}")
            result = await self._call_thinking_api(client, "/chat/completions", payload, ctx, 60, stream=False)
            elapsed_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Thinking model query completed in {elapsed_time:.3f} seconds")
            logger.info(f"Response data: {result}")
            content = result.get("choices", [])[0].get("message", {}).get("content", "")
            if extract_json:
                content = self._extract_json_from_response(content, force_first_element)
            return content
        except Exception as e:
            logger.error(f"Error querying thinking model: {e}")
            raise e

    def generate_document_outline(self, document=None, template_id: str = None) -> str:
        """
        Generate a markdown outline of the document or template.
        """
        outline = []
        if document:
            sections = document.get("template", {}).get("sections", [])
        elif hasattr(self, 'current_document') and self.current_document:
            sections = self.current_document.get("template", {}).get("sections", [])
        elif template_id and template_id in self.templates:
            sections = self.templates[template_id].get("sections", [])
        else:
            return "No document or valid template available to generate outline."
        def build_outline(sections, level=1):
            for sec in sections:
                prefix = "#" * level
                title = sec.get("title", "Untitled")
                outline.append(f"{prefix} {title}")
                if sec.get("children"):
                    build_outline(sec["children"], level + 1)
        build_outline(sections)
        return "\n".join(outline)

    async def handle_remove_section(self, chat_id, params):
        """Handle removing a section from the document."""
        section_name = params.get("section_name", "")
        yield {"think": f"Attempting to remove section: {section_name}...\n"}
        if chat_id not in self.documents:
            yield {"think": "No document found. Please generate a document first."}
            yield {"result": "# No Document Found\n\nNo document available. Use 'write_full_doc' first."}
            return
        document = self.documents[chat_id]
        section, path = self.find_section_by_name(document, section_name)
        if not section:
            yield {"think": f"Section '{section_name}' not found.\n"}
            yield {"result": "# Section Not Found\n\nSection '{section_name}' not found. Use 'list_sections' to view available sections."}
            return
        if "/" in path:
            parent_path, _ = path.rsplit("/", 1)
            parent_section = self._find_section_by_path(document["template"]["sections"], parent_path)
            if parent_section and parent_section.get("children"):
                idx = next((i for i, child in enumerate(parent_section["children"]) if child["name"] == section["name"]), -1)
                if idx >= 0:
                    parent_section["children"].pop(idx)
                    yield {"think": "Removed section from parent's children.\n"}
        else:
            idx = next((i for i, s in enumerate(document["template"]["sections"]) if s["name"] == section["name"]), -1)
            if idx >= 0:
                document["template"]["sections"].pop(idx)
                yield {"think": "Removed top-level section from template.\n"}
        if section["name"] in document["sections"]:
            del document["sections"][section["name"]]
            yield {"think": "Removed section content.\n"}
        def remove_children(sec):
            if sec.get("children"):
                for child in sec["children"]:
                    if child["name"] in document["sections"]:
                        del document["sections"][child["name"]]
                        yield {"think": f"Removed child section: {child['name']}\n"}
                    yield from remove_children(child)
        for msg in remove_children(section):
            yield {"think": msg}
        document["updated_at"] = time.time()
        if self.valves.DOCUMENT_STORAGE == "file":
            self.save_document(chat_id, document)
            yield {"think": "Saved updated document to file storage.\n"}
        output = [f"# Section Removed\n\nRemoved section '**{section['title']}**' from the document.\n", "\n## Updated Document Structure\n\n", self.list_document_sections(document)]
        yield {"result": "\n".join(output)}

    def _find_section_by_path(self, sections, path):
        """Find a section by its path in the hierarchy."""
        path_parts = path.split("/")
        current_sections = sections
        for i, part in enumerate(path_parts):
            found = False
            for sec in current_sections:
                if sec["name"] == part:
                    if i == len(path_parts) - 1:
                        return sec
                    elif sec.get("children"):
                        current_sections = sec["children"]
                        found = True
                        break
            if not found:
                return None
        return None

    async def _get_rag_context(self, ctx, topic, section_prompt):
        """
        Retrieve context from RAG systems for the given topic and section prompt.
        Returns context text as a string.
        """

        if not self.valves.ENABLE_RAG:
            return ""

        context_texts = []

        logger.info(f"RAG is enabled - using knowledge base to enhance document generation")

        try:
            # Use vector database retrieval if available
            if ctx["request"] and hasattr(ctx["request"], "app") and hasattr(ctx["request"].app, "state"):
                app_state = ctx["request"].app.state

                # Prepare the query combining topic and section prompt
                # query = f"{topic} {section_prompt}"
                query = f"{section_prompt}"

                logger.info(f"query: {query}")
                
                # Determine which search method to use
                results = {}
                # logger.info(f"RAG_HYBRID_SEARCH: {self.valves.RAG_HYBRID_SEARCH}")
                # logger.info(f"rf: {app_state.rf}")
                # logger.info(f"EMBEDDING_FUNCTION: {app_state.EMBEDDING_FUNCTION}")
                # logger.info(f"queries: {query}")
                # logger.info(f"collection_names: {collection_names}")
                # logger.info(f"user: {user}")
                # logger.info(f"used RAG_RESULT_COUNT: {self.valves.RAG_RESULT_COUNT if self.valves.RAG_RESULT_COUNT else app_state.config.RAG_RESULT_COUNT}")
                # logger.info(f"used RELEVANCE_THRESHOLD: {self.valves.RELEVANCE_THRESHOLD if self.valves.RELEVANCE_THRESHOLD else app_state.config.RELEVANCE_THRESHOLD}")
                # logger.info(f"used RAG_WEB_SEARCH: {self.valves.RAG_WEB_SEARCH}")
                logger.info(f"used query_collection_with_hybrid_search?: {self.valves.RAG_HYBRID_SEARCH and hasattr(app_state, 'rf') and app_state.rf}, relevance threshold: {self.valves.RELEVANCE_THRESHOLD if self.valves.RELEVANCE_THRESHOLD else app_state.config.RELEVANCE_THRESHOLD}")
                # logger.info(f"query_embedding: {app_state.EMBEDDING_FUNCTION(query, user=user)}")
                
                k = self.valves.RAG_RESULT_COUNT if self.valves.RAG_RESULT_COUNT else app_state.config.RAG_RESULT_COUNT

                relevance_threshold = self.valves.RELEVANCE_THRESHOLD if self.valves.RELEVANCE_THRESHOLD else app_state.config.RELEVANCE_THRESHOLD
                if self.valves.RAG_HYBRID_SEARCH:
                    results = query_collection_with_hybrid_search(
                        queries=[query],
                        collection_names=ctx["collection_names"],
                        embedding_function=lambda query: app_state.EMBEDDING_FUNCTION(
                            query, user=ctx["user"]
                        ),
                        k=k,
                        reranking_function=app_state.rf,
                        r=relevance_threshold,
                    )
                    # Flatten distances
                    flattened_distances = [dist for sublist in results["distances"] for dist in sublist]
                    # Flatten documents
                    context_texts = [doc for doc_list in results["documents"] for doc in doc_list]
                    # Filter context_texts based on r
                    context_texts = [doc for i, doc in enumerate(context_texts) if flattened_distances[i] >= relevance_threshold]

                    logger.info(f"flattened_distances: {[e for e in flattened_distances]}")
                else:
                    results = query_collection(
                        queries=[query],
                        collection_names=ctx["collection_names"],
                        embedding_function=lambda query: app_state.EMBEDDING_FUNCTION(
                            query, user=ctx["user"]
                        ),
                        k=k,
                    )
                    context_texts = [doc for doc_list in results["documents"] for doc in doc_list]

                logger.info(f"distances: {results['distances']}")
            
            # Use web search if enabled
            if self.valves.RAG_WEB_SEARCH:
                query = f"{topic} {section_prompt}"
                web_results = []
                
                # Use DuckDuckGo as default, adjust if you want to support other engines
                if self.valves.RAG_WEB_SEARCH_ENGINE == "duckduckgo":
                    web_results = await search_duckduckgo(
                        query=query, 
                        max_results=self.valves.RAG_RESULT_COUNT
                    )
                
                # Load content from search results
                if web_results:
                    web_loader = get_web_loader(ssl_verify=True)
                    for result in web_results:
                        if hasattr(result, "url") and result.url:
                            try:
                                content = await web_loader.load_content(result.url)
                                if content:
                                    context_texts.append(f"From {result.url}:\n{content[:1000]}...")
                            except Exception as e:
                                logger.warning(f"Failed to load content from {result.url}: {e}")
            
            # Combine all context
            combined_context = "\n\n---\n\n".join(context_texts)
            
            # Log the amount of context retrieved
            logger.info(f"Retrieved {len(context_texts)} context pieces for RAG")
            
            return (combined_context, len(context_texts))

        except Exception as e:
            logger.error(f"Error retrieving RAG context: {e}")
            return ""
        
    def extract_collection_names(self, data):
        """
        Extracts all collection UUID strings from the provided dictionary.
        
        The function looks for collections in two ways:
        1. In the top-level "files" list, it finds any file whose "type" is "collection"
            and adds its "id" (if present) to the result.
        2. It also checks each file's nested "files" list and, for each nested file,
            if the "meta" dictionary contains a "collection_name", that value is added.
            
        Args:
            data (dict): The dictionary to search for collection UUIDs.
            
        Returns:
            list: A list of unique collection UUID strings.
        """
        collections = set()
        if not data or not data.get("files"):
            return collections
        
        # Loop through top-level files, if any.
        for file in data.get("files", []):
            # Check if this file is a collection.
            if file.get("type") == "collection":
                # Add the collection id if available.
                if "id" in file:
                    collections.add(file["id"])
                
                # Check nested files for a collection_name in their meta data.
                for nested_file in file.get("files", []):
                    meta = nested_file.get("meta", {})
                    if "collection_name" in meta:
                        collections.add(meta["collection_name"])
        
        return list(collections)
    
    def handle_action_chunk(self, data, ctx):
        """
        Handle a chunk of an action.
        """
        if "think" in data:
            return data["think"]
        elif "result" in data:
            ctx["result"] += data["result"]
            return ""

    async def check_for_document_edits(self, messages, chat_id, ctx):
        """
        Check if the most recent assistant message contains an edited document and update it if necessary.
        Returns True if a document was edited and updated, False otherwise.
        """
        if not messages:
            return False
        
        # We should only process edited messages
        assistant_messages = [m for m in messages if m.get("role") == "assistant"]
        if not assistant_messages:
            return False
            
        last_message = assistant_messages[-1]

                    
        logger.info("Checking for document changes")
        
        # Check if we have an existing document for this chat
        if chat_id not in self.documents:
            logger.info("No existing document found for this chat")
            return False
        
        # Extract the document from the edited message
        edited_doc_text = self.extract_document_from_message(last_message.get("content", ""))
        if not edited_doc_text:
            logger.info("No document found in edited message")
            return False

        found_full_doc = False
        for line in edited_doc_text.split("\n"):
            logger.info(f"Line: {line}")
            if re.search(r'.*\{#fulldoc\}.*', line, re.MULTILINE):
                found_full_doc = True
                break

        if not found_full_doc:
            logger.info("No full document found in edited message")
            return False
        
        logger.info(f"Extracted document text: {json.dumps(edited_doc_text, indent=2)}")
        # Parse the edited document
        try:
            parsed_document = await self.parse_document_from_text(edited_doc_text, ctx)
            logger.info(f"Parsed document: {json.dumps(parsed_document, indent=2)}")
            if not parsed_document:
                logger.warning("Failed to parse edited document")
                return False
            
            # Update the existing document
            existing_document = self.documents[chat_id]
            updated = self.update_document_from_edit(existing_document, parsed_document)
            logger.info(f"Updated: {updated}")
            if updated:
                logger.info("Document updated successfully from edits")
                self.save_document(chat_id, existing_document)
                return True
                
        except Exception as e:
            logger.error(f"Error processing document edits: {str(e)}")
            
        return False
        
    def extract_document_from_message(self, message_content):
        """
        Extract the document portion from a message.
        Returns the document text or None if no document is found.
        """

        message_content = re.sub(r'<details.*?</details>', '', message_content, re.MULTILINE)

        if not message_content or "# " not in message_content:
            return None
            
        # Try to find document content between markdown section markers
        sections = re.split(r'\n\s*---+\s*\n', message_content)
        
        # Find the section that looks most like a document
        candidate_sections = []
        for section in sections:
            if not section:
                continue
            section = section.strip()
            # Check if section has multiple markdown headings
            headings = re.findall(r'^#+\s+.+', section, re.MULTILINE)
            if len(headings) > 1:
                # Score based on number of headings and section length
                score = len(headings) * 10 + len(section)
                candidate_sections.append((section, score))
        
        if candidate_sections:
            # Return the section with the highest score
            return max(candidate_sections, key=lambda x: x[1])[0]
        
        # If no clear multi-heading section, check if the whole message is a document
        if len(re.findall(r'^#+\s+.+', message_content, re.MULTILINE)) > 1:
            return message_content
            
        return None
        
    async def parse_document_from_text(self, document_text, ctx):
        """
        Parse a document from text, extracting its structure and content.
        Returns a dictionary with the parsed document or None if parsing fails.
        """
        if not document_text:
            return None
            
        # Initialize document structure
        parsed_doc = {
            "sections": {},
            "structure": [],
            "updated_at": time.time()
        }
        
        # Extract the title/topic (first heading)
        title_match = re.search(r'^#\s+(.+?)(?:\s*\{#.*?\})?\s*$', document_text, re.MULTILINE)
        if title_match:
            parsed_doc["title"] = title_match.group(1).strip()
        else:
            parsed_doc["title"] = "Edited Document"
            
        # Parse section content
        current_section = None
        section_content = []
        section_stack = []
        
        for line in document_text.split('\n'):
            # Check if line is a heading
            heading_match = re.match(r'^(#+)\s+(.+?)(?:\s*\{#(.*?)\})?\s*$', line)
            if heading_match:
                # If we were processing a section, save its content
                if current_section:
                    section_text = '\n'.join(section_content).strip()
                    if section_text:
                        parsed_doc["sections"][current_section] = section_text
                
                # Process the new section
                heading_level = len(heading_match.group(1))
                section_title = heading_match.group(2).strip()
                
                # Get section name from id if available, otherwise generate from title
                if heading_match.group(3):
                    section_name = heading_match.group(3).strip()
                else:
                    section_name = re.sub(r'[^a-z0-9_]', '', section_title.lower().replace(' ', '_'))
                    # Ensure uniqueness
                    counter = 1
                    base_name = section_name
                    while section_name in parsed_doc["sections"]:
                        section_name = f"{base_name}_{counter}"
                        counter += 1
                
                # Update section stack based on heading level
                while section_stack and section_stack[-1]["level"] >= heading_level:
                    section_stack.pop()
                
                # Create new section node
                section_node = {
                    "name": section_name,
                    "title": section_title,
                    "level": heading_level,
                    "children": []
                }
                
                # Add to structure
                if not section_stack:
                    parsed_doc["structure"].append(section_node)
                else:
                    section_stack[-1]["children"].append(section_node)
                
                # Update stack
                section_stack.append(section_node)
                
                current_section = section_name
                section_content = []
                section_level = heading_level
            elif current_section:
                section_content.append(line)
        
        # Save the last section's content
        if current_section and section_content:
            section_text = '\n'.join(section_content).strip()
            if section_text:
                parsed_doc["sections"][current_section] = section_text
        
        # Clean up structure for export (remove level attribute)
        def clean_structure(nodes):
            result = []
            for node in nodes:
                cleaned = {
                    "name": node["name"],
                    "title": node["title"],
                    "children": clean_structure(node.get("children", []))
                }
                result.append(cleaned)
            return result
        
        parsed_doc["structure"] = clean_structure(parsed_doc["structure"])
        return parsed_doc

    def update_document_from_edit(self, existing_document, parsed_document):
        """
        Update an existing document with changes from an edited version.
        Returns True if changes were made, False otherwise.
        """
        if not existing_document or not parsed_document:
            return False
            
        changes_made = False
        
        # Update document topic if changed
        if parsed_document.get("title") and parsed_document["title"] != existing_document.get("title"):
            existing_document["title"] = parsed_document["title"]
            changes_made = True
            logger.info(f"Updated document title to: {parsed_document['title']}")
        
        # Build a map of existing sections by normalized title for better matching
        existing_sections_by_title = {}
        existing_sections_by_name = {}
        
        def map_sections(sections):
            for section in sections:
                normalized_title = re.sub(r'\s+', ' ', section["title"].lower().strip())
                existing_sections_by_title[normalized_title] = section
                existing_sections_by_name[section["name"]] = section
                if section.get("children"):
                    map_sections(section["children"])
        
        map_sections(existing_document["template"]["sections"])

        #find the children of the root section
        #sections_to_update =  parsed_document["structure"][0]["children"]
        sections_to_update =  parsed_document["structure"]
        logger.info(f"Sections to update: {json.dumps(sections_to_update, indent=2)}")
        
        # Update section content based on matching by name or normalized title
        for section_name, section_content in parsed_document["sections"].items():
            # First try to match by section name
            if section_name in existing_document["sections"]:
                if existing_document["sections"][section_name] != section_content:
                    existing_document["sections"][section_name] = section_content
                    changes_made = True
                    logger.info(f"Updated content for section: {section_name}")
            else:
                # Try to match by title in the parsed structure
                matched = False
                for struct_section in sections_to_update:
                    if self._find_section_by_name(struct_section, section_name):
                        section_info = self._find_section_by_name(struct_section, section_name)
                        normalized_title = re.sub(r'\s+', ' ', section_info["title"].lower().strip())
                        
                        if normalized_title in existing_sections_by_title:
                            existing_section = existing_sections_by_title[normalized_title]
                            existing_name = existing_section["name"]
                            
                            if existing_name in existing_document["sections"]:
                                if existing_document["sections"][existing_name] != section_content:
                                    existing_document["sections"][existing_name] = section_content
                                    changes_made = True
                                    logger.info(f"Updated content for title-matched section: {existing_name}")
                                    matched = True
                                    break
                
                # If no match found, this might be a new section
                if not matched:
                    # Only add if we find it in the structure
                    for struct_section in sections_to_update:
                        section_info = self._find_section_by_name(struct_section, section_name)
                        if section_info:
                            # This appears to be a new section
                            existing_document["sections"][section_name] = section_content
                            changes_made = True
                            logger.info(f"Added new section: {section_name}")
                            break
        
        # Update structure if needed (more complex)
        if sections_to_update:
            # Check if structure has changed significantly
            structure_changed = self._structure_has_significant_changes(
                existing_document["template"]["sections"],
                sections_to_update
            )
            
            if structure_changed:
                logger.info("Significant structure changes detected")
                # Merge structure changes
                self.merge_document_structure(
                    existing_document["template"]["sections"], 
                    sections_to_update
                )
                changes_made = True
        
        if changes_made:
            existing_document["updated_at"] = time.time()
            
        return changes_made

    def _find_section_by_name(self, section, name):
        """Helper to find a section by name in a nested structure"""
        logger.info(f"Finding section by name: {name} in section: {section}")
        if section["name"] == name:
            return section
        
        for child in section.get("children", []):
            result = self._find_section_by_name(child, name)
            if result:
                return result
        
        return None

    def _structure_has_significant_changes(self, existing_sections, new_sections):
        """
        Determine if the new structure has significant changes compared to existing.
        This is more nuanced than a direct comparison.
        """
        # Compare section counts - a different number of sections is significant
        if len(existing_sections) != len(new_sections):
            return True
        
        # Create normalized title sets for quick comparison
        existing_titles = set()
        new_titles = set()
        
        def collect_titles(sections, title_set):
            for section in sections:
                normalized_title = re.sub(r'\s+', ' ', section["title"].lower().strip())
                title_set.add(normalized_title)
                if section.get("children"):
                    collect_titles(section.get("children", []), title_set)
        
        collect_titles(existing_sections, existing_titles)
        collect_titles(new_sections, new_titles)
        
        # If the sets of titles differ by more than 20%, consider it significant
        added = len(new_titles - existing_titles)
        removed = len(existing_titles - new_titles)
        max_count = max(len(existing_titles), len(new_titles))
        
        if max_count > 0 and (added + removed) / max_count > 0.2:
            return True
        
        # Otherwise, do a deeper check on matching titles and hierarchy
        return self._check_hierarchy_changes(existing_sections, new_sections)

    def _check_hierarchy_changes(self, existing_sections, new_sections):
        """Check if the hierarchy structure has changed significantly"""
        # Map sections by normalized title
        existing_map = {}
        new_map = {}
        
        def map_section_paths(sections, path_map, current_path=""):
            for i, section in enumerate(sections):
                normalized_title = re.sub(r'\s+', ' ', section["title"].lower().strip())
                section_path = f"{current_path}/{normalized_title}" if current_path else normalized_title
                path_map[normalized_title] = section_path
                
                if section.get("children"):
                    map_section_paths(section["children"], path_map, section_path)
        
        map_section_paths(existing_sections, existing_map)
        map_section_paths(new_sections, new_map)
        
        # Check for sections that exist in both but have different paths
        common_titles = set(existing_map.keys()) & set(new_map.keys())
        different_paths = 0
        
        for title in common_titles:
            if existing_map[title] != new_map[title]:
                different_paths += 1
        
        # If more than 30% of common sections changed hierarchy, it's significant
        if common_titles and different_paths / len(common_titles) > 0.3:
            return True
        
        return False

    def merge_document_structure(self, existing_sections, new_sections):
        """
        Intelligently merge new section structure into existing structure,
        preserving section metadata like prompts, types, etc.
        """
        # First, build a mapping of existing sections
        existing_by_name = {}
        existing_by_title = {}
        
        def map_sections(sections):
            for section in sections:
                existing_by_name[section["name"]] = section
                normalized_title = re.sub(r'\s+', ' ', section["title"].lower().strip())
                existing_by_title[normalized_title] = section
                if section.get("children"):
                    map_sections(section["children"])
        
        map_sections(existing_sections)
        
        # Now create the updated structure
        updated_sections = []
        
        for new_section in new_sections:
            normalized_title = re.sub(r'\s+', ' ', new_section["title"].lower().strip())
            
            # Try to match by name first, then by normalized title
            if new_section["name"] in existing_by_name:
                # Found by name - update the existing section
                existing = existing_by_name[new_section["name"]]
                updated = self._update_section_preserving_metadata(existing, new_section)
                updated_sections.append(updated)
            elif normalized_title in existing_by_title:
                # Found by title - update the existing section
                existing = existing_by_title[normalized_title]
                updated = self._update_section_preserving_metadata(existing, new_section)
                updated_sections.append(updated)
            else:
                # This is a new section - create with defaults
                new_with_metadata = {
                    "name": new_section["name"],
                    "title": new_section["title"],
                    "prompt": f"Write content about {new_section['title']}",
                    "sectionType": "free_text",
                    "children": []
                }
                
                # Recursively process children
                if new_section.get("children"):
                    new_with_metadata["children"] = []
                    for child in new_section["children"]:
                        # Recursively process child sections
                        child_sections = self.merge_document_structure([child], [])
                        if child_sections:
                            new_with_metadata["children"].extend(child_sections)
                
                updated_sections.append(new_with_metadata)
        
        # If this is a recursive call, return the merged sections
        if not existing_sections or len(existing_sections) == 0:
            return updated_sections
        
        # Otherwise, update in place
        existing_sections.clear()
        existing_sections.extend(updated_sections)
        return None

    def _update_section_preserving_metadata(self, existing, new_section):
        """Update a section while preserving its metadata"""
        # Start with a copy of the existing section to preserve all metadata
        updated = existing.copy()
        
        # Update the title if it changed
        if existing["title"] != new_section["title"]:
            updated["title"] = new_section["title"]
        
        # Process children recursively
        if new_section.get("children"):
            if "children" not in updated:
                updated["children"] = []
            
            # Merge the children
            child_sections = []
            self.merge_document_structure(updated["children"], new_section["children"])
        
        return updated

    async def handle_import_document(self, chat_id, params, ctx):
        """
        Import a document from text pasted by the user.
        Creates a document structure based on the provided markdown content.
        """
        document_text = params.get("document_text", "")
        
        if not document_text or "# " not in document_text:
            yield {"think": "No valid document text provided. Document must contain markdown headings.\n"}
            yield {"result": "# Invalid Document Format\n\nPlease provide a document with proper markdown headings (# for title, ## for sections, etc.)."}
            return
        
        yield {"think": "Parsing document from user-provided text...\n"}
        
        try:
            # Parse the document text into a structured format
            parsed_document = await self.parse_document_from_text(document_text, ctx)
            
            if not parsed_document:
                yield {"think": "Failed to parse document.\n"}
                yield {"result": "# Parsing Error\n\nCould not parse the provided document. Please check the format and try again."}
                return
                
            # Create a new document from the parsed content
            new_document = await self.create_document_from_parsed(parsed_document, ctx)
            
            # Save the document
            self.documents[chat_id] = new_document
            
            if self.valves.DOCUMENT_STORAGE == "file":
                self.save_document(chat_id, new_document)
                yield {"think": f"Document saved to file storage for chat {chat_id}.\n"}
                
            # Generate output message
            section_count = len(new_document["sections"])
            top_level_count = len(new_document["template"]["sections"])
            
            yield {"think": f"Created document with title '{new_document['title']}' containing {section_count} sections.\n"}
            yield {"result": f"# Document Imported Successfully\n\nCreated document on **{new_document['title']}** with {section_count} sections.\n\n## Document Structure\n\n{self.list_document_sections(new_document)}\n\n---\n\nYou can now use the document commands like `print_document` to view the full document."}
            
        except Exception as e:
            logger.error(f"Error importing document: {e}")
            yield {"think": f"Error: {e}\n"}
            yield {"result": f"# Import Failed\n\nAn error occurred while importing the document: {e}"}

    async def create_document_from_parsed(self, parsed_document, ctx):
        """
        Convert a parsed document structure into a complete document format.
        """
        # Extract basic information
        title = parsed_document.get("title", "Imported Document")
        sections = parsed_document.get("sections", {})
        structure = parsed_document.get("structure", [])
        
        # Create template sections with proper metadata
        template_sections = self.convert_structure_to_template(structure)
        
        # Create the document object
        document = {
            "title": title,
            "topic": title,
            "template": {
                "name": f"{title} Document",
                "description": f"Document about {title} imported from text",
                "sections": template_sections #template_sections[0]["children"]
            },
            "model_id": self.valves.MODEL_ID,
            "sections": sections,
            "created_at": time.time(),
            "updated_at": time.time()
        }

        # Generate metadata
        metadata = await self.generate_document_metadata(parsed_document, ctx)
        logger.info(f"Generated metadata: {metadata}")
        
        document["metadata"] = metadata
        for meta in self.document_metadata:
            document[meta["name"]] = metadata.get(meta["name"], "Not provided")
        return document

    def convert_structure_to_template(self, structure):
        """
        Convert the parsed document structure into template sections with proper metadata.
        """
        template_sections = []
        
        def process_section(section):
            # Create a template section with metadata
            template_section = {
                "name": section["name"],
                "title": section["title"],
                "prompt": f"Write content about {section['title']}",
                "sectionType": "free_text",
                "children": []
            }
            
            # Process children recursively
            if section.get("children"):
                for child in section["children"]:
                    template_section["children"].append(process_section(child))
                    
            return template_section
        
        # Process all top-level sections
        for section in structure:
            template_sections.append(process_section(section))
        
        return template_sections
    
    async def generate_document_metadata(self,  parsed_document, ctx):
        """
        Query the thinking model to generate metadata for a document.
        Returns a JSON object with metadata fields.
        """
        # Extract document content for context
        title = parsed_document.get("title", "")
        topic = parsed_document.get("topic", "")
        sections_content = "\n\n".join([
            f"## {section_name}\n{content}" 
            for section_name, content in parsed_document.get("sections", {}).items()
        ])
        logger.info(f"Sections content: {parsed_document}")

        tmp_document = {
            "title": title,
            "topic": title,
            "template": {
                "sections": parsed_document.get("structure", [])
            },
            "sections": parsed_document.get("sections", {})
        }
        #logger.info(f"Tmp document: {tmp_document}")

        document_sections = self.generate_document_outline(tmp_document)
        # logger.info(f"Document sections: {document_sections}")

        system_prompt = (
            f"{self.system_prompts['generic_init']}. "
            f"You are analyzing a document titled '{title}' to generate metadata. "
            f"Based on this content, generate the following metadata fields: "
            f"{json.dumps(self.document_metadata, indent=2)}. "
            f"Format your response as a JSON object containing these fields. "
            f"Do not include any other text or comments in your response.\n"
        )

        user_prompt = (
            f"Document Title: {title}. "
            f"this is a sample of the document content: "
            f"{sections_content[:2000]}... [content truncated]\n. "
            f"this is the structure of the document: "
            f"{document_sections}\n"
        )
        
        try:
            # logger.info("Querying thinking model to generate document metadata, system prompt: {system_prompt}, user prompt: {user_prompt}")
            
            response = await self._query_thinking_model(ctx["client"], system_prompt, user_prompt, extract_json=True, ctx=ctx)
            
            if not response or not isinstance(response, dict):
                logger.warning("Failed to generate document metadata, using defaults")
                # Provide fallback values
                return {
                    "keywords": ["document", topic.lower()],
                    "summary": f"Document about {topic}",
                    "audience": "General",
                    "purpose": "Information",
                    "category": "General",
                    "difficulty_level": "Intermediate"
                }
                
            logger.info(f"Generated document metadata: {response}")
            return response
            
        except Exception as e:
            logger.error(f"Error generating document metadata: {e}")
            # Return minimal metadata in case of error
            return {
                "keywords": ["document", topic.lower()],
                "summary": f"Document about {topic}"
            }