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
from typing import Dict, List, Optional, Any, Union, Generator, Iterator
from fastapi import Request
from pprint import pprint

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
        SMART_MODEL: bool = Field(
            default=True,
            description="Use the thinking model for action detection (true) or use hardcoded patterns (false)"
        )
        TEMPERATURE: float = Field(
            default=0.7,
            description="Temperature for generation"
        )
        MAX_TOKENS: int = Field(
            default=1000,
            description="Maximum tokens per section generation"
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
                        "sectionType": "generate",
                        "children": []
                    },
                    {
                        "name": "body",
                        "title": "Main Content",
                        "prompt": "Provide detailed information about {topic}. Include key facts, analysis, and insights.",
                        "sectionType": "generate",
                        "children": [
                            {
                                "name": "body_part1",
                                "title": "Key Concepts",
                                "prompt": "Explain the key concepts related to {topic}.",
                                "sectionType": "generate",
                                "children": []
                            },
                            {
                                "name": "body_part2",
                                "title": "Analysis",
                                "prompt": "Provide analysis and interpretation of {topic}.",
                                "sectionType": "generate",
                                "children": []
                            }
                        ]
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion summarizing the key points about {topic}.",
                        "sectionType": "generate",
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
                        "sectionType": "generate"
                    },
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an academic introduction for research on {topic}. Include research questions.",
                        "sectionType": "generate"
                    },
                    {
                        "name": "literature",
                        "title": "Literature Review",
                        "prompt": "Provide a literature review for research on {topic}. Reference key studies and findings.",
                        "sectionType": "generate"
                    },
                    {
                        "name": "methodology",
                        "title": "Methodology",
                        "prompt": "Describe a research methodology for studying {topic}.",
                        "sectionType": "generate"
                    },
                    {
                        "name": "results",
                        "title": "Results and Discussion",
                        "prompt": "Present hypothetical results and discussion for research on {topic}.",
                        "sectionType": "generate"
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion for academic research on {topic}. Include limitations and future directions.",
                        "sectionType": "generate"
                    },
                    {
                        "name": "references",
                        "title": "References",
                        "prompt": "Generate a sample reference list for research on {topic}. Use proper academic citation format.",
                        "sectionType": "generate"
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
                        "sectionType": "generate",
                        "children": []
                    },
                    {
                        "name": "disclaimer",
                        "title": "Disclaimer",
                        "sectionType": "fixed",
                        "template_text": "This document is for informational purposes only. The information provided is not legal or professional advice and should not be relied upon as such. Consult with appropriate experts before making decisions based on this content.",
                        "children": []
                    },
                    {
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
                    },
                    {
                        "name": "main_content",
                        "title": "Main Content",
                        "prompt": "Write detailed content about {topic}.",
                        "sectionType": "generate",
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
                        "sectionType": "generate"
                    }
                ]
            },
            "test_advanced_fields": {
                "name": "Test Advanced Fields",
                "description": "Test advanced fields",
                "sections": [
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an introduction about {topic}. Provide context and background.",
                        "sectionType": "generate",
                        "children": []
                    },
                    {
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
        }

        # Load templates from Git repositories and local storage
        self.load_git_templates()
        self._load_local_templates()

        logger.info(f"Templates loaded: {self.templates}")

        self.system_prompts = {
            "generic_init": (
                "You are an AI assistant helping with document generation. The document is in markdown format. Do not add comments or opinions."
            ),
            "no_sections": (
                "Do not include any additional sections marked by the '#' symbol, headings, or subsections."
            ),
            "no_conclusions": (
                "The response must not contain phrases like 'finally', 'in conclusion', 'in summary', or 'in the end'."
            )
        }

        # Define available actions
        self.actions = {
            "write_full_doc": {
                "description": "Generate a complete document",
                "requires": [
                    {"name": "topic", "type": "string", "description": "The topic of the document"},
                    {"name": "template_id", "type": "enum", "values": list(self.templates.keys()), "description": "The template to use for document generation"}
                ]
            },
            "edit_section": {
                "description": "Edit or regenerate a specific section",
                "requires": [
                    {"name": "section_name", "type": "string", "description": "The name of the section to edit"},
                    {"name": "instructions", "type": "string", "description": "Editing instructions for the section"}
                ]
            },
            "add_section": {
                "description": "Add a new section or sub-section to the document",
                "requires": [
                    {"name": "section_title", "type": "string", "description": "The title for the new section"},
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
                "requires": []
            },
            "expand_section": {
                "description": "Expand a specific section with more details",
                "requires": [
                    {"name": "section_name", "type": "string", "description": "The name of the section to expand"},
                    {"name": "instructions", "type": "string", "description": "Instructions for expanding the section"}
                ]
            },
            "rewrite_section": {
                "description": "Rewrite a section in a different style",
                "requires": [
                    {"name": "section_name", "type": "string", "description": "The name of the section to rewrite"},
                    {"name": "style", "type": "string", "description": "The style to rewrite the section in"}
                ]
            },
            "list_sections": {
                "description": "List all sections in the current document",
                "requires": []
            },
            "generate_tags": {
                "description": "Generate tags for the chat content",
                "requires": []
            },
            "list_actions": {
                "description": "List all available actions with their descriptions",
                "requires": []
            },
            "get_all_templates": {
                "description": "Display all available document templates",
                "requires": []
            },
            "generate_template": {
                "description": "Generate a document template from a markdown document",
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
                "requires": [
                    {"name": "template_id", "type": "string", "description": "The ID of the template to delete"}
                ]
            },
            "get_template_json": {
                "description": "Get the JSON structure of a document template",
                "requires": [
                    {"name": "template_id", "type": "string", "description": "The ID of the template to retrieve"}
                ]
            },
            "update_template": {
                "description": "Update a template with the provided JSON",
                "requires": [
                    {"name": "template_id", "type": "string", "description": "The ID of the template to update"},
                    {"name": "template_json", "type": "string", "description": "The JSON structure for the template"}
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

    async def pipe(self, body: dict, __event_emitter__=None, __metadata__=None) -> Union[str, Generator, Iterator]:
        """
        Main pipeline handler that dispatches to specific action handlers.
        It extracts user messages, detects the action to perform, and then routes the request accordingly.
        """
        user_message = ""
        model_id = ""
        messages = []
        chat_id = __metadata__.get("chat_id", "default_chat")

        logger.info(f"Pipe body: {body}")

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

        # If using file storage, try to load an existing document
        if self.valves.DOCUMENT_STORAGE == "file":
            loaded_document = self.load_document(chat_id)
            if loaded_document:
                self.documents[chat_id] = loaded_document
                self.document = loaded_document
                logger.info(f"Loaded existing document for chat {chat_id}")

        # Validate API key
        if not self.valves.API_KEY:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Error: API key not configured", "done": True}
                })
            yield "Error: API key not configured. Please set the API_KEY valve."
            return

        # Prepare request headers
        headers = self._get_api_headers(self.valves.API_KEY)

        try:
            # Start thinking process
            yield "<think>\n"
            yield "🧠 Analyzing your request...\n"

            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            timeout = httpx.Timeout(self.valves.TIMEOUT, connect=5.0)
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                # Detect the action
                action_result = await self.detect_action(client, user_message, headers)
                if "error" in action_result:
                    yield f"Error detecting action: {action_result['error']}\n"
                    yield "Falling back to default document generation...\n"
                    action = "write_full_doc"
                    params = {"topic": user_message, "template_id": self.valves.TEMPLATE_NAME}
                else:
                    action = action_result["action"]
                    params = action_result.get("parameters", {})
                    confidence = action_result.get("confidence", 0.0)
                    yield f"Detected action: {self.actions[action]['description']} ({action}) (confidence: {confidence:.2f})\n"
                    yield f"Parameters: {json.dumps(params, indent=2)}\n\n"

                # Dispatch to appropriate handler
                if action == "generate_tags":
                    result = await self.handle_generate_tags(client, user_message, headers)
                    logger.info(f"Generated tags: '{result}'")
                    yield result
                    return
                elif action == "write_full_doc":
                    async for chunk in self.handle_write_full_doc(client, chat_id, user_message, model_id, params, headers, __event_emitter__):
                        yield chunk
                elif action == "edit_section":
                    async for chunk in self.handle_edit_section(client, chat_id, params, headers, __event_emitter__):
                        yield chunk
                elif action == "add_section":
                    async for chunk in self.handle_add_section(client, chat_id, user_message, params, headers, __event_emitter__):
                        yield chunk
                elif action == "remove_section":
                    async for chunk in self.handle_remove_section(chat_id, params):
                        yield chunk
                elif action == "list_sections":
                    async for chunk in self.handle_list_sections(chat_id):
                        yield chunk
                elif action == "summarize_doc":
                    async for chunk in self.handle_summarize_doc(client, chat_id, params, headers, __event_emitter__):
                        yield chunk
                elif action == "expand_section":
                    async for chunk in self.handle_expand_section(client, chat_id, params, headers, __event_emitter__):
                        yield chunk
                elif action == "rewrite_section":
                    async for chunk in self.handle_rewrite_section(client, chat_id, params, headers, __event_emitter__):
                        yield chunk
                elif action == "list_actions":
                    async for chunk in self.handle_list_actions():
                        yield chunk
                elif action == "get_all_templates":
                    async for chunk in self.handle_get_all_templates():
                        yield chunk
                elif action == "generate_template":
                    async for chunk in self.generate_template_from_markdown(client, user_message, headers):
                        yield chunk
                elif action == "print_document":
                    async for chunk in self.handle_print_document(chat_id):
                        yield chunk
                elif action == "delete_template":
                    async for chunk in self.handle_delete_template(params):
                        yield chunk
                elif action == "get_template_json":
                    async for chunk in self.handle_get_template_json(params):
                        yield chunk
                elif action == "update_template":
                    async for chunk in self.handle_update_template(params):
                        yield chunk
                else:
                    yield f"Action '{action}' not implemented or recognized. Showing available actions instead.\n"
                    yield "</think>\n\n"
                    result = await self.handle_list_actions()
                    yield result

            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Operation completed", "done": True}
                })
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Document operation error: {error_msg}")
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
            yield "</think>\n\n"

    async def detect_action(self, client, user_message, headers=None):
        """
        Detect the action requested by the user.
        Returns a dict with keys: action, parameters, and confidence.
        """
        lower_msg = user_message.lower()
        # Check for tag generation request
        if "generate 1-3 broad tags categorizing the main themes of the chat history" in user_message:
            return {"action": "generate_tags", "parameters": {}, "confidence": 1.0}

        # Help actions
        help_patterns = ["help", "what can you do", "show actions", "available actions", "list actions"]
        if any(pattern in lower_msg for pattern in help_patterns):
            return {"action": "list_actions", "parameters": {}, "confidence": 1.0}

        # Template list actions
        template_patterns = ["show templates", "list templates", "available templates", "get templates", "what templates"]
        if any(pattern in lower_msg for pattern in template_patterns):
            return {"action": "get_all_templates", "parameters": {}, "confidence": 1.0}

        # Template generation from markdown
        template_gen_patterns = ["generate template", "create template", "make template", "extract template"]
        if any(pattern in lower_msg for pattern in template_gen_patterns) and '#' in user_message:
            return {"action": "generate_template", "parameters": {"markdown": user_message}, "confidence": 0.9}

        # If SMART_MODEL is false, use basic pattern matching
        if not self.valves.SMART_MODEL:
            if any(p in lower_msg for p in ["write", "create", "generate", "make a document", "prepare a document"]):
                template_id = self.valves.TEMPLATE_NAME
                for tid in self.templates.keys():
                    if tid.lower() in lower_msg:
                        template_id = tid
                        break
                return {"action": "write_full_doc", "parameters": {"topic": user_message, "template_id": template_id}, "confidence": 0.8}
            elif any(p in lower_msg for p in ["edit", "change", "modify", "update", "revise"]):
                match = re.search(r'(edit|change|modify|update|revise)\s+(\w+)\s+section', lower_msg)
                section_name = match.group(2) if match else "introduction"
                return {"action": "edit_section", "parameters": {"section_name": section_name, "instructions": user_message}, "confidence": 0.7}
            elif any(p in lower_msg for p in ["add", "insert", "include", "create section"]):
                match = re.search(r'add\s+(?:a\s+)?(?:section|part)\s+(?:called|named|titled)?\s+["\']?([^"\']+)["\']?', lower_msg)
                section_title = match.group(1) if match else "New Section"
                return {"action": "add_section", "parameters": {"section_title": section_title, "instructions": user_message}, "confidence": 0.7}
            elif any(p in lower_msg for p in ["list sections", "show sections", "what sections"]):
                return {"action": "list_sections", "parameters": {}, "confidence": 0.9}
            elif any(p in lower_msg for p in ["summarize", "summary", "brief overview"]):
                return {"action": "summarize_doc", "parameters": {}, "confidence": 0.8}
            elif any(p in lower_msg for p in ["expand", "elaborate", "more details", "extend"]):
                match = re.search(r'(expand|elaborate|extend)\s+(\w+)\s+section', lower_msg)
                section_name = match.group(2) if match else ""
                return {"action": "expand_section", "parameters": {"section_name": section_name, "instructions": user_message}, "confidence": 0.7}
            elif any(p in lower_msg for p in ["rewrite", "rephrase", "different style", "new style"]):
                match = re.search(r'(rewrite|rephrase)\s+(\w+)\s+section', lower_msg)
                style_match = re.search(r'(?:in|with|using)\s+(?:a\s+)?(\w+)\s+style', lower_msg)
                section_name = match.group(2) if match else ""
                style = style_match.group(1) if style_match else "formal"
                return {"action": "rewrite_section", "parameters": {"section_name": section_name, "style": style}, "confidence": 0.7}
            else:
                return {"action": "list_actions", "parameters": {}, "confidence": 0.6}

        # Use the thinking model if SMART_MODEL is enabled
        logger.info(f"Detecting action using thinking model for message: {user_message}")
        if not headers:
            headers = self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY)

        # Build a system prompt with available actions and requirements
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
            "\nAlways respond with a JSON array containing objects with these fields:\n"
            "- action: The action name\n- parameters: Relevant parameters (e.g., topic, section_name, etc.)\n- confidence: Your confidence (0.0-1.0)\n"
            "\nExample response:\n[{{\"action\": \"write_full_doc\", \"parameters\": {{\"topic\": \"artificial intelligence\", \"template_id\": \"basic\"}}, \"confidence\": 0.9}}]"
        )
        user_prompt = f"What document action should I take based on this request: '{user_message}'?"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return await self._query_thinking_model(client, system_prompt, user_prompt, extract_json=True, force_first_element=True)
            except Exception as e:
                logger.error(f"Error detecting action on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                example_json = '[{"action": "write_full_doc", "parameters": {"topic": "example topic", "template_id": "basic"}, "confidence": 0.9}]'
                user_prompt = (f"I need a valid JSON response for this request: '{user_message}'\n"
                               f"Return ONLY a JSON array with a single action object, like this example:\n{example_json}\n"
                               f"Choose one of these actions: {', '.join(self.actions.keys())}")
                await asyncio.sleep(1)
        return {"action": "list_actions", "parameters": {}, "confidence": 0.5}

    def _extract_json_from_response(self, content, force_first_element=False):
        """
        Extract JSON from a response that may contain <think> tags or other text.
        """
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
            if force_first_element and isinstance(data, list) and data:
                return data[0]
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e} with content: {content}")
            raise

    async def _stream_chat_response(self, client, url, payload, headers, timeout, max_retries=2):
        """
        Helper method to stream chat responses with retry logic.
        Yields each JSON-decoded data chunk.
        """
        for attempt in range(max_retries + 1):
            try:
                async with client.stream("POST", url, json=payload, headers=headers, timeout=timeout) as response:
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
                        chunk_str = chunk.decode('utf-8')
                        for line in chunk_str.split('\n'):
                            if line.startswith('data: ') and line.strip() != "data: [DONE]":
                                try:
                                    data = json.loads(line[6:])
                                    yield data
                                except json.JSONDecodeError as e:
                                    logger.error(f"JSON parse error in stream: {e}")
                return
            except httpx.TimeoutException:
                if attempt < max_retries:
                    logger.error("Request timed out, retrying...")
                    await asyncio.sleep(5)
                    continue
                else:
                    yield {"error": f"Request timed out after {timeout} seconds"}
                    return
            except Exception as e:
                if attempt < max_retries:
                    logger.error(f"Exception in stream: {str(e)}, retrying...")
                    await asyncio.sleep(5)
                    continue
                else:
                    yield {"error": str(e)}
                    return

    async def generate_section(self, client, model_id, section, topic, headers):
        """
        Generate a single section using the API with streaming.
        """
        section_type = section.get("sectionType", "generate")
        system_prompt = self.system_prompts["generic_init"]

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
                    "You are given a field prompt and a topic. Generate a plain text value for the field without markdown formatting. "
                    f"This is for the section \"{section['title']}\". Provide only the field content."
                )
                field_value = ""
                try:
                    payload = {
                        "model": model_id,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": field_prompt}
                        ],
                        "temperature": self.valves.TEMPERATURE,
                        "max_tokens": min(200, self.valves.MAX_TOKENS),
                        "stream": False,
                    }
                    response = await client.post(
                        f"{self.valves.API_HOST}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=self.valves.REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    response_data = response.json()
                    field_value = response_data.get("choices", [])[0].get("message", {}).get("content", "").strip()
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
        prompt = section["prompt"].format(topic=topic)
        system_prompt += (
            f"{self.system_prompts['no_sections']}"
            f"You are generating content for a document on the topic: '{topic}'.\n"
            f"Document structure:\n{document_structure}\n\n"
            f"Generate a single paragraph for the section \"{section['title']}\". "
            f"{self.system_prompts['no_conclusions']}"
        )
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,
        }
        logger.info(f"Generating section with payload: {payload['messages']}")
        timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)
        content_buffer = ""
        async for data in self._stream_chat_response(client, f"{self.valves.API_HOST}/chat/completions", payload, headers, timeout):
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
                        if "children" in s and s["children"]:
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
            if "children" in s and s["children"]:
                result = self.find_section_in_template(s["children"], section_name)
                if result:
                    return result
        return None

    async def edit_document_section(self, client, model_id, section_name, document, instructions, headers):
        """Edit an existing section of the document."""
        section_content = document["sections"].get(section_name, "")
        section_info = self.find_section_in_template(document["template"]["sections"], section_name)
        system_prompt = self.system_prompts["generic_init"]
        document_structure = self._generate_document_outline(section_info)
        topic = document.get("topic", "Unknown")
        system_prompt += (
            f"{self.system_prompts['no_sections']}"
            f"You are editing a document on the topic: '{topic}'.\n"
            f"Document structure:\n{document_structure}\n\n"
            f"The response must be a single paragraph for the section and adhere to the style. "
            f"{self.system_prompts['no_conclusions']}"
        )
        if not section_info:
            yield {"error": f"Section '{section_name}' not found in the document"}
            return
        prompt = (f"Edit the following document section according to these instructions: {instructions}\n\n"
                  f"Section: {section_info['title']}\n\n"
                  f"Current content:\n{section_content}\n\n"
                  "Provide the complete edited section:")
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,
        }
        logger.info(f"Editing section with payload: {payload['messages']}")
        timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)
        content_buffer = ""
        async for data in self._stream_chat_response(client, f"{self.valves.API_HOST}/chat/completions", payload, headers, timeout):
            if "error" in data:
                yield {"error": data["error"]}
                break
            if 'choices' in data and data["choices"]:
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    piece = delta["content"]
                    content_buffer += piece
                    yield {"content_piece": piece}
        yield {"content": content_buffer, "status": "complete"}

    async def add_document_section(self, client, model_id, document, section_title, instructions, headers):
        """Add a new section to the document."""
        base_name = re.sub(r'[^a-z0-9]', '', section_title.lower())
        section_name = base_name
        counter = 1
        while section_name in document["sections"]:
            section_name = f"{base_name}{counter}"
            counter += 1
        system_prompt = self.system_prompts["generic_init"]
        topic = document.get("topic", "Unknown")
        document_structure = self.generate_document_outline(document)
        system_prompt += (
            f"{self.system_prompts['no_sections']}"
            f"You are creating a new section for a document on the topic: '{topic}'.\n"
            f"Document structure:\n{document_structure}\n\n"
            f"Generate a new paragraph for the section \"{section_title}\". "
            f"{self.system_prompts['no_conclusions']}"
        )
        prompt = f"Create a new document section titled \"{section_title}\" according to these instructions: {instructions}"
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,
        }
        logger.info(f"Adding section with payload: {payload['messages']}")
        timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)
        content_buffer = ""
        async for data in self._stream_chat_response(client, f"{self.valves.API_HOST}/chat/completions", payload, headers, timeout):
            if "error" in data:
                yield {"error": data["error"]}
                break
            if 'choices' in data and data["choices"]:
                delta = data["choices"][0].get("delta", {})
                if "content" in delta:
                    piece = delta["content"]
                    content_buffer += piece
                    yield {"content_piece": piece, "section_name": section_name, "section_title": section_title}
        new_section = {
            "name": section_name,
            "title": section_title,
            "prompt": f"Write content for '{section_title}' about {{topic}}. {instructions}",
            "children": []
        }
        yield {
            "content": content_buffer,
            "status": "complete",
            "section_name": section_name,
            "section_title": section_title,
            "section_metadata": new_section
        }

    async def handle_write_full_doc(self, client, chat_id, user_message, model_id, params, headers, __event_emitter__=None):
        """Handle generating a full document."""
        topic = params.get("topic", user_message).strip()
        template_id = params.get("template_id", self.valves.TEMPLATE_NAME)
        if not topic:
            yield "Error: No topic provided. Please specify a topic for the document."
            yield "</think>\n\n"
            return
        if template_id not in self.templates:
            template_id = self.valves.TEMPLATE_NAME
        template = self.templates[template_id]
        llm_model_id = model_id.split(":", 1)[0] if ":" in model_id else model_id
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Generating document on '{topic}'", "done": False}
            })
        yield f"Generating full document on '{topic}' using template: {template['name']}\n\n"
        document = {
            "topic": topic,
            "template": template,
            "model_id": llm_model_id,
            "sections": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }
        async for chunk in self.generate_document(client, llm_model_id, template, topic, headers):
            yield chunk
        if hasattr(self, 'current_document') and self.current_document:
            document = self.current_document
            self.current_document = None
        self.documents[chat_id] = document
        self.save_document(chat_id, document)
        yield "\nCompiling final document...\n"
        yield "</think>\n\n"
        yield self.format_document(document)

    async def handle_edit_section(self, client, chat_id, params, headers, __event_emitter__=None):
        """Handle editing a specific section."""
        section_name = params.get("section_name", "")
        instructions = params.get("instructions", "Improve this section")
        if chat_id not in self.documents:
            yield "No document found to edit. Please generate a document first."
            yield "</think>\n\n"
            return
        document = self.documents[chat_id]
        llm_model_id = document["model_id"]
        section, section_path = self.find_section_by_name(document, section_name)
        if not section or section["name"] not in document["sections"]:
            yield f"Section '{section_name}' not found in the document.\n"
            yield "Available sections:\n"
            yield self.list_document_sections(document)
            yield "</think>\n\n"
            return
        section_title = section["title"]
        section_name = section["name"]
        yield f"Editing section: {section_title} ({section_name}) with instructions: {instructions}\n\n"
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Editing section: {section_title}", "done": False}
            })
        section_content = ""
        async for chunk in self.edit_document_section(client, llm_model_id, section_name, document, instructions, headers):
            if "error" in chunk:
                yield f"Error editing section: {chunk['error']}\n"
                break
            elif "content_piece" in chunk:
                section_content += chunk["content_piece"]
                yield chunk["content_piece"]
            elif "status" in chunk and chunk["status"] == "complete":
                document["sections"][section_name] = chunk.get("content", section_content)
                document["updated_at"] = time.time()
                self.save_document(chat_id, document)
                yield f"\nCompleted editing section: {section_title}\n"
        yield "</think>\n\n"
        yield self.format_document(document, highlight_section=section_name)

    async def determine_section_position(self, client, document, section_title, position_info, headers):
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
            position_data = await self._query_thinking_model(client, system_prompt, user_prompt, extract_json=True, force_first_element=True)
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
            is_sub_section = position_data.get("is_nested_sub_section", False)
            if reference_section == "end" or not reference_section:
                logger.info("Adding section to the end of document")
                document["template"]["sections"].append(section_metadata)
                return True

            def find_and_insert(sections, parent_name=None):
                for i, sec in enumerate(sections):
                    if sec["name"] == reference_section:
                        logger.info(f"Found reference section: {sec['name']} at level {parent_name or 'root'}")
                        if is_sub_section:
                            sec.setdefault("children", []).append(section_metadata)
                            return True
                        if parent_section and parent_name != parent_section:
                            continue
                        if position == "before":
                            sections.insert(i, section_metadata)
                        else:
                            sections.insert(i + 1, section_metadata)
                        return True
                    if sec.get("children"):
                        if find_and_insert(sec["children"], sec["name"]):
                            return True
                if parent_section == parent_name:
                    sections.append(section_metadata)
                    return True
                return False

            success = find_and_insert(document["template"]["sections"])
            if not success:
                logger.warning(f"Reference section {reference_section} not found. Adding at end.")
                document["template"]["sections"].append(section_metadata)
            return True
        except Exception as e:
            logger.error(f"Error inserting section: {e}")
            try:
                document["template"]["sections"].append(section_metadata)
                return True
            except Exception:
                return False

    async def handle_add_section(self, client, chat_id, user_message, params, headers, __event_emitter__=None):
        """Handle adding a new section to the document."""
        section_title = params.get("section_title", "New Section")
        instructions = params.get("instructions", "Add content for this section")
        yield f"Adding new section: {section_title} with instructions: {instructions}\n\n"
        yield f"Analyzing where to place the section based on: {user_message}\n"
        try:
            position_data = await self.determine_section_position(client, self.documents[chat_id], section_title, user_message, headers)
            yield f"Position determined: {json.dumps(position_data, indent=2)}\n\n"
        except Exception as e:
            logger.error(f"Error in position determination: {e}")
            position_data = {"parent_section": None, "reference_section": "end", "position": "after", "explanation": "Default due to error"}
            yield f"Error determining position: {e}. Adding to the end of document.\n\n"
        if __event_emitter__:
            await __event_emitter__({
                "type": "status",
                "data": {"description": f"Adding section: {section_title}", "done": False}
            })
        section_content = ""
        section_name = None
        section_metadata = None
        async for chunk in self.add_document_section(client, self.documents[chat_id]["model_id"], self.documents[chat_id], section_title, instructions, headers):
            if "error" in chunk:
                yield f"Error adding section: {chunk['error']}\n"
                break
            elif "content_piece" in chunk:
                section_content += chunk["content_piece"]
                section_name = chunk.get("section_name", section_name)
                yield f"{chunk['content_piece']}"
            elif "status" in chunk and chunk["status"] == "complete":
                section_content = chunk.get("content", section_content)
                section_name = chunk.get("section_name", section_name)
                section_metadata = chunk.get("section_metadata")
                if section_name and section_metadata:
                    self.documents[chat_id]["sections"][section_name] = section_content
                    success = self.insert_section_into_document(self.documents[chat_id], section_metadata, position_data)
                    if success:
                        self.documents[chat_id]["updated_at"] = time.time()
                        self.save_document(chat_id, self.documents[chat_id])
                        yield f"\nCompleted adding section: {section_title}\nSection placed {position_data.get('position', 'after')} {position_data.get('reference_section', 'the end')}\n"
                    else:
                        yield "\nError positioning the section. Added at the end instead.\n"
                        self.documents[chat_id]["template"]["sections"].append(section_metadata)
                        self.documents[chat_id]["updated_at"] = time.time()
                        self.save_document(chat_id, self.documents[chat_id])
        yield "</think>\n\n"
        yield self.format_document(self.documents[chat_id], highlight_section=section_name)

    async def handle_list_sections(self, chat_id):
        """Handle listing all sections in the document."""
        if chat_id not in self.documents:
            yield "No document found. Please generate a document first."
            yield "</think>\n\n"
            return
        document = self.documents[chat_id]
        yield f"Listing sections for document on topic: {document['topic']}\n\n"
        yield "</think>\n\n"
        yield self.list_document_sections(document)

    async def handle_summarize_doc(self, client, chat_id, params, headers, __event_emitter__=None):
        """Handle summarizing the existing document."""
        yield "Summarize document action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Document Summary\n\nThis feature is coming soon!"

    async def handle_expand_section(self, client, chat_id, params, headers, __event_emitter__=None):
        """Handle expanding a specific section with more details."""
        yield "Expand section action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Section Expansion\n\nThis feature is coming soon!"

    async def handle_rewrite_section(self, client, chat_id, params, headers, __event_emitter__=None):
        """Handle rewriting a section in a different style."""
        yield "Rewrite section action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Section Rewriting\n\nThis feature is coming soon!"

    def format_document(self, document, highlight_section=None, include_metadata=False):
        """
        Format the document as markdown for display with support for nested sections.
        """
        output = []
        if "topic" in document:
            output.append(f"# {document['topic']}\n")
        else:
            output.append("# Generated Document\n")
        toc_entries = []

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
            sec_type = sec.get("sectionType", "generate")
            if sec_key in document["sections"] or sec_type in ["empty", "toc"]:
                if sec_type == "toc":
                    output.append(f"{heading} {sec['title']} (Table of Contents{' - Updated' if sec_key == highlight_section else ''})\n")
                    for entry in toc_entries:
                        output.append(f"{'  ' * entry['indent']}[{entry['title']}](#{entry['name']})\n")
                    output.append("\n")
                # elif sec_type == "empty":
                #     # output.append(f"{heading} {sec['title']} ({'Empty Section - Updated' if sec_key == highlight_section else 'Empty Section'})\n\n")
                #     output.append(f"{heading} {sec['title']} ({'Empty Section - Updated' if sec_key == highlight_section else 'Empty Section'})\n\n")
                else:
                    # label = "(Fixed Section" if sec_type == "fixed" else ("(Template Section" if sec_type == "populate" else "(Updated)" if sec_key == highlight_section else "") + ")"
                    label = ""
                    output.append(f"{heading} {sec['title']} {label}\u007b#{sec['name']}\u007d\n\n")
                    if sec_type != "empty" and sec_key in document["sections"]:
                        output.append(f"{document['sections'][sec_key]}\n\n")
                if sec.get("children"):
                    for child in sec["children"]:
                        format_section(child, level + 1)

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

    async def generate_document(self, client, llm_model_id, template, topic, headers):
        """Generate a complete document with nested sections."""
        document = {
            "topic": topic,
            "template": template,
            "model_id": llm_model_id,
            "sections": {},
            "created_at": time.time(),
            "updated_at": time.time()
        }

        async def generate_section_recursive(sec, parent_path=""):
            sec_name = sec["title"]
            sec_key = sec["name"]
            yield f"Generating section: {sec_name}...\n"
            section_content = ""
            has_error = False
            async for chunk in self.generate_section(client, llm_model_id, sec, topic, headers):
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

        for sec in template["sections"]:
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
        output = [f"# Document Sections for: {document.get('topic', 'Untitled')}\n"]
        def format_listing(sections, indent=0):
            for sec in sections:
                sec_name = sec["name"]
                sec_title = sec["title"]
                prefix = "  " * indent + "- "
                if sec_name in document["sections"]:
                    word_count = len(document["sections"][sec_name].split())
                    output.append(f"{prefix}**{sec_title}** ({word_count} words) {sec_name}\n")
                else:
                    output.append(f"{prefix}{sec_title} (not generated)\n")
                if sec.get("children"):
                    format_listing(sec["children"], indent + 1)
        format_listing(document["template"]["sections"])
        output.append(f"\n**Topic:** {document.get('topic', 'Unspecified')}\n")
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
                timestamped_filename = f"{os.path.splitext(template_filename)[0]}_{timestamp}.json"
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

    async def handle_generate_tags(self, client, message, headers):
        """Handle generating tags for the chat content."""
        logger.info("Generating tags for chat content")
        system_prompt = (
            "You are an expert at categorizing content. Generate appropriate tags that summarize the main themes discussed. "
            "Return only a JSON object with a 'tags' field."
        )
        try:
            content = await self._query_thinking_model(client, system_prompt, message, extract_json=True)
            logger.info(f"Generated tags: {content}")
            return content
        except Exception as e:
            logger.error(f"Error generating tags: {e}")
            return '{"tags": ["General"]}'

    async def handle_list_actions(self):
        """Handle listing all available actions."""
        yield "Listing all available actions for document generation...\n"
        yield "</think>\n\n"
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
        yield "\n".join(output)

    async def handle_get_all_templates(self):
        """Handle displaying all available templates."""
        yield "Retrieving all available document templates...\n"
        yield "</think>\n\n"
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
        yield "\n".join(output)

    async def generate_template_from_markdown(self, client, user_message, headers=None):
        """
        Generate a template from a markdown document provided by the user.
        """
        yield "Analyzing markdown document to generate a template...\n"
        markdown_lines = user_message.strip().split('\n')
        if not any(line.startswith('#') for line in markdown_lines):
            yield "Error: No markdown headings found. Please provide a document with headings starting with #.\n"
            yield "</think>\n\n"
            yield "# Template Generation Failed\n\nNo headings found in the document."
            return
        section_tree = []
        current_sections = {0: None}
        section_content = {}
        current_section_type = "generate"
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
                            yield f"Warning: Error parsing populate fields in '{title}'.\n"
                    else:
                        current_section_type = "generate"
                        current_section_populate_fields = []
                    line = f"# {title}"
                else:
                    current_section_type = "generate"
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
        yield f"Extracted {len(all_sections)} sections from the document.\n"
        for sec in all_sections:
            sec_name = sec['name']
            sec_title = sec['title']
            content = "\n".join(section_content.get(sec_name, []))
            sec_type = sec.get('sectionType', 'generate')
            if sec_type == "fixed":
                sec['template_text'] = content
                sec['prompt'] = f"Fixed section: {sec_title}"
                yield f"Configured fixed section: '{sec_title}'\n"
            elif sec_type == "populate":
                sec['template_text'] = content
                sec['prompt'] = f"Populate template for {sec_title} with field values"
                yield f"Configured populate section: '{sec_title}' with {len(sec.get('populateFields', []))} fields\n"
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
                    yield f"Generated prompt for '{sec_title}': {prompt_text}\n"
                except Exception as e:
                    logger.error(f"Error generating prompt for section {sec_name}: {e}")
                    sec['prompt'] = f"Explain {sec_title.lower()} for {{topic}}."
                    yield f"Error generating prompt for '{sec_title}': {e}\n"
        yield "Generating template metadata...\n"
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
            yield f"Generated template name: {template_name}\n"
            yield f"Generated description: {template_description}\n"
        except Exception as e:
            logger.error(f"Error generating template metadata: {e}")
            yield f"Error generating template metadata: {e}\n"
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
            yield f"Template saved to {template_path}\n"
        except Exception as e:
            logger.error(f"Error saving template: {e}")
            yield f"Error saving template: {e}\n"
        yield "</think>\n\n"
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
        yield "\n".join(output)

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
            yield "No document found. Please generate a document first."
            yield "</think>\n\n"
            yield "# No Document Found\n\nPlease create a document using the 'write_full_doc' action."
            return
        document = self.documents[chat_id]
        yield f"Retrieving document on topic: {document.get('topic', 'Untitled')}\n"
        yield "</think>\n\n"
        yield self.format_document(document, include_metadata=True)

    async def handle_delete_template(self, params):
        """Handle deleting a template."""
        template_id = params.get("template_id", "")
        yield f"Attempting to delete template: {template_id}...\n"
        if template_id not in self.templates:
            yield f"Error: Template '{template_id}' not found.\n"
            yield "</think>\n\n"
            yield f"# Template Not Found\n\nNo template with ID '{template_id}' was found."
            return
        builtin_templates = ["basic", "research", "advanced", "custom"]
        if template_id in builtin_templates:
            yield f"Error: Cannot delete built-in template '{template_id}'.\n"
            yield "</think>\n\n"
            yield f"# Cannot Delete Built-in Template\n\nTemplate '{template_id}' is built-in and cannot be deleted."
            return
        template_name = self.templates[template_id].get("name", template_id)
        file_deleted = False
        template_files = glob.glob(f"/app/backend/data/doc_generator_pipe/templates/{template_id}.json")
        for file_path in template_files:
            try:
                os.remove(file_path)
                file_deleted = True
                yield f"Deleted template file: {file_path}\n"
            except Exception as e:
                yield f"Warning: Failed to delete template file {file_path}: {e}\n"
        try:
            del self.templates[template_id]
            yield f"Removed template '{template_id}' from memory.\n"
            yield "</think>\n\n"
            yield f"# Template Deleted\n\nTemplate '{template_name}' (ID: {template_id}) has been successfully deleted."
            if file_deleted:
                yield "\n\nTemplate file was also removed."
        except Exception as e:
            yield f"Error removing template from memory: {e}\n"
            yield "</think>\n\n"
            yield f"# Template Deletion Error\n\nAn error occurred while deleting template '{template_name}' (ID: {template_id}): {e}"

    async def handle_get_template_json(self, params):
        """Handle getting a template's JSON structure."""
        template_id = params.get("template_id", "")
        yield f"Retrieving JSON for template: {template_id}...\n"
        if template_id not in self.templates:
            yield f"Error: Template '{template_id}' not found.\n"
            yield "</think>\n\n"
            yield f"# Template Not Found\n\nNo template with ID '{template_id}' was found."
            return
        template = self.templates[template_id]
        template_json = json.dumps(template, indent=2)
        yield f"Successfully retrieved template JSON for '{template_id}'.\n"
        yield "</think>\n\n"
        yield f"# Template: {template['name']} (ID: {template_id})\n\n"
        if "description" in template:
            yield f"{template['description']}\n\n"
        yield "## Template JSON Structure\n\n"
        yield f"```json\n{template_json}\n```\n\n"
        yield "You can use this JSON to inspect or create a new template."

    async def handle_update_template(self, params):
        """Handle updating a template with provided JSON."""
        template_id = params.get("template_id", "")
        template_json_str = params.get("template_json", "")
        yield f"Attempting to update template: {template_id}...\n"
        if template_id not in self.templates:
            yield f"Template '{template_id}' does not exist. This will create a new template.\n"
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
                yield f"Warning: Updating built-in template '{template_id}'. It will be updated in memory only.\n"
            self.templates[template_id] = template_data
            yield f"Updated template '{template_id}' in memory.\n"
            if template_id not in builtin_templates:
                try:
                    file_path = f"/app/backend/data/doc_generator_pipe/templates/{template_id}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(template_data, f, ensure_ascii=False, indent=2)
                    yield f"Saved template to file: {file_path}\n"
                except Exception as e:
                    yield f"Warning: Failed to save template to file: {e}\n"
            yield "</think>\n\n"
            yield f"# Template Updated Successfully\n\nThe template '{template_data['name']}' (ID: {template_id}) has been updated.\n\n"
            yield f"## Template Summary\n\n**Name:** {template_data['name']}\n"
            if "description" in template_data:
                yield f"**Description:** {template_data['description']}\n"
            yield f"**Number of Sections:** {len(template_data['sections'])}\n\n"
            yield "### Sections\n\n"
            for sec in template_data["sections"]:
                yield f"- {sec['title']}\n"
        except json.JSONDecodeError as e:
            yield f"Error: Invalid JSON format: {e}\n"
            yield "</think>\n\n"
            yield f"# Template Update Failed\n\nInvalid JSON: {e}\n\nPlease correct and try again."
        except ValueError as e:
            yield f"Error: {e}\n"
            yield "</think>\n\n"
            yield f"# Template Update Failed\n\n{e}\n\nPlease correct and try again."
        except Exception as e:
            yield f"Error updating template: {e}\n"
            yield "</think>\n\n"
            yield f"# Template Update Failed\n\nAn unexpected error occurred: {e}"

    async def _query_thinking_model(self, client, system_prompt, user_prompt, extract_json=False, temperature=0.3, force_first_element=False, **kwargs):
        """
        Query the thinking model and clean the response.
        """
        start_time = asyncio.get_event_loop().time()
        try:
            payload = {
                "model": self.valves.THINKING_MODEL_ID,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ] + kwargs.get("messages", []),
                "temperature": temperature,
                "max_tokens": kwargs.get("max_tokens", 4000)
            }
            logger.info(f"Payload Messages: {payload['messages']}")
            response = await client.post(
                f"{self.valves.THINKING_API_HOST}/chat/completions",
                headers=self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY),
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()
            elapsed_time = asyncio.get_event_loop().time() - start_time
            logger.info(f"Thinking model query completed in {elapsed_time:.3f} seconds")
            logger.info(f"Response data: {response_data}")
            content = response_data.get("choices", [])[0].get("message", {}).get("content", "")
            if extract_json:
                content = self._extract_json_from_response(content, force_first_element)
            return content
        except Exception as e:
            logger.error(f"Error querying thinking model: {e}")
            raise

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
        yield f"Attempting to remove section: {section_name}...\n"
        if chat_id not in self.documents:
            yield "No document found. Please generate a document first."
            yield "</think>\n\n"
            yield "# No Document Found\n\nNo document available. Use 'write_full_doc' first."
            return
        document = self.documents[chat_id]
        section, path = self.find_section_by_name(document, section_name)
        if not section:
            yield f"Section '{section_name}' not found.\n"
            yield "</think>\n\n"
            yield f"# Section Not Found\n\nSection '{section_name}' not found. Use 'list_sections' to view available sections."
            return
        if "/" in path:
            parent_path, _ = path.rsplit("/", 1)
            parent_section = self._find_section_by_path(document["template"]["sections"], parent_path)
            if parent_section and parent_section.get("children"):
                idx = next((i for i, child in enumerate(parent_section["children"]) if child["name"] == section["name"]), -1)
                if idx >= 0:
                    parent_section["children"].pop(idx)
                    yield "Removed section from parent's children.\n"
        else:
            idx = next((i for i, s in enumerate(document["template"]["sections"]) if s["name"] == section["name"]), -1)
            if idx >= 0:
                document["template"]["sections"].pop(idx)
                yield "Removed top-level section from template.\n"
        if section["name"] in document["sections"]:
            del document["sections"][section["name"]]
            yield "Removed section content.\n"
        def remove_children(sec):
            if sec.get("children"):
                for child in sec["children"]:
                    if child["name"] in document["sections"]:
                        del document["sections"][child["name"]]
                        yield f"Removed child section: {child['name']}\n"
                    yield from remove_children(child)
        for msg in remove_children(section):
            yield msg
        document["updated_at"] = time.time()
        if self.valves.DOCUMENT_STORAGE == "file":
            self.save_document(chat_id, document)
            yield "Saved updated document to file storage.\n"
        yield "</think>\n\n"
        output = [f"# Section Removed\n\nRemoved section '**{section['title']}**' from the document.\n", "\n## Updated Document Structure\n\n", self.list_document_sections(document)]
        yield "\n".join(output)

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
