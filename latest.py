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

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Generator, Iterator
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
from fastapi import Request
from pprint import pprint


# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Pipe:
    class Valves(BaseModel):
        API_HOST: str = Field(
            default="https://api.openai.com/v1",
            description="API Host URL for content generation",
        )
        API_KEY: str = Field(default="", description="API Key for the model provider")
        MODEL_ID: str = Field(
            default="gpt-3.5-turbo",
            description="Model ID to use for generation (comma-separated for multiple models)",
        )
        THINKING_API_HOST: str = Field(
            default="https://api.openai.com/v1",
            description="API Host URL for the thinking model",
        )
        THINKING_API_KEY: str = Field(
            default="",
            description="API Key for the thinking model (defaults to API_KEY if empty)",
        )
        THINKING_MODEL_ID: str = Field(
            default="gpt-4o",
            description="Model ID to use for thinking/action detection",
        )
        TEMPERATURE: float = Field(
            default=0.7, description="Temperature for generation"
        )
        MAX_TOKENS: int = Field(
            default=1000, description="Maximum tokens per section generation"
        )
        TIMEOUT: int = Field(
            default=300, description="Timeout in seconds for API requests"
        )
        REQUEST_TIMEOUT: float = Field(
            default=60.0, description="Timeout for individual request chunks"
        )
        DOCUMENT_STORAGE: str = Field(
            default="memory",
            description="Storage mechanism for documents (memory or file)",
        )
        TEMPLATE_NAME: str = Field(
            default="basic",
            description="The fallback template to use for document generation, if no template is specified / found",
        )
        GIT_TEMPLATE_REPOS: str = Field(
            default="",
            description="Comma-separated list of Git repositories containing additional templates (e.g., https://github.com/user/repo1.git,https://github.com/user/repo2.git)",
        )

        class Config:
            # Additional configuration to improve schema documentation
            json_schema_extra = {
                "examples": [
                    {
                        "TEMPLATE_NAME": "technical_documentation",
                        "GIT_TEMPLATE_REPOS": "https://github.com/user/doc-templates.git,https://github.com/organization/templates-collection.git",
                    }
                ]
            }

    def __init__(self):
        self.valves = self.Valves()
        # Store for documents in memory
        self.documents = {}

        # Create necessary directories
        os.makedirs("/app/backend/data/doc_generator_pipe", exist_ok=True)
        os.makedirs("/app/backend/data/doc_generator_pipe/git", exist_ok=True)
        os.makedirs("/app/backend/data/doc_generator_pipe/templates", exist_ok=True)

        # Embedded templates with support for nested sections
        self.templates = {
            "basic": {
                "name": "Basic Document",
                "description": "A simple document with introduction, body, and conclusion",
                "sections": [
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an introduction about {topic}. Provide context and background.",
                        "children": [],  # No children for this section
                    },
                    {
                        "name": "body",
                        "title": "Main Content",
                        "prompt": "Provide detailed information about {topic}. Include key facts, analysis, and insights.",
                        "children": [
                            {
                                "name": "body_part1",
                                "title": "Key Concepts",
                                "prompt": "Explain the key concepts related to {topic}.",
                                "children": [],
                            },
                            {
                                "name": "body_part2",
                                "title": "Analysis",
                                "prompt": "Provide analysis and interpretation of {topic}.",
                                "children": [],
                            },
                        ],
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion summarizing the key points about {topic}.",
                        "children": [],
                    },
                ],
            },
            "research": {
                "name": "Research Paper",
                "description": "Academic research paper structure",
                "sections": [
                    {
                        "name": "abstract",
                        "title": "Abstract",
                        "prompt": "Write a concise abstract for a research paper about {topic}.",
                    },
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an academic introduction for research on {topic}. Include research questions.",
                    },
                    {
                        "name": "literature",
                        "title": "Literature Review",
                        "prompt": "Provide a literature review for research on {topic}. Reference key studies and findings.",
                    },
                    {
                        "name": "methodology",
                        "title": "Methodology",
                        "prompt": "Describe a research methodology for studying {topic}.",
                    },
                    {
                        "name": "results",
                        "title": "Results and Discussion",
                        "prompt": "Present hypothetical results and discussion for research on {topic}.",
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion for academic research on {topic}. Include limitations and future directions.",
                    },
                    {
                        "name": "references",
                        "title": "References",
                        "prompt": "Generate a sample reference list for research on {topic}. Use proper academic citation format.",
                    },
                ],
            },
            "custom": {
                "name": "Custom Document",
                "description": "User-defined document structure",
                "sections": [
                    {
                        "name": "sample_section",
                        "title": "Sample Section",
                        "prompt": "Generate a sample section for a document on {topic}.",
                    }
                ],
            },
        }

        # Load templates from Git repositories
        self.load_git_templates()

        logger.info(f"Templates: {self.templates}")

        # Define available actions
        self.actions = {
            "write_full_doc": {
                "description": "Generate a complete document",
                "requires": [
                    {
                        "name": "topic",
                        "type": "string",
                        "description": "The topic of the document",
                    },
                    {
                        "name": "template_id",
                        "type": "enum",
                        "values": list(self.templates.keys()),
                        "description": "The template to use for document generation",
                    },
                ],
            },
            "edit_section": {
                "description": "Edit or regenerate a specific section",
                "requires": [
                    {
                        "name": "section_name",
                        "type": "string",
                        "description": "The name of the section to edit",
                    },
                    {
                        "name": "instructions",
                        "type": "string",
                        "description": "The instructions for the section",
                    },
                ],
            },
            "add_section": {
                "description": "Add a new section to the document",
                "requires": [
                    {
                        "name": "section_title",
                        "type": "string",
                        "description": "The title for the new section",
                    },
                    {
                        "name": "instructions",
                        "type": "string",
                        "description": "The instructions for the section content",
                    },
                    # {"name": "position", "type": "string", "description": "Positioning information like 'after introduction' or 'before conclusion'"}
                ],
            },
            "summarize_doc": {
                "description": "Summarize the existing document",
                "requires": [
                    {
                        "name": "chat_id",
                        type: "string",
                        "description": "The chat ID of the document",
                    }
                ],
            },
            "expand_section": {
                "description": "Expand a specific section with more details",
                "requires": [
                    {
                        "name": "section_name",
                        type: "string",
                        "description": "The name of the section to expand",
                    },
                    {
                        "name": "instructions",
                        type: "string",
                        "description": "The instructions for the section",
                    },
                ],
            },
            "rewrite_section": {
                "description": "Rewrite a section in a different style",
                "requires": [
                    {
                        "name": "section_name",
                        type: "string",
                        "description": "The name of the section to rewrite",
                    },
                    {
                        "name": "style",
                        type: "string",
                        "description": "The style to rewrite the section in",
                    },
                ],
            },
            "list_sections": {
                "description": "List all sections in the current document",
                "requires": [
                    {
                        "name": "chat_id",
                        type: "string",
                        "description": "The chat ID of the document",
                    }
                ],
            },
            "create_summary": {
                "description": "Create an outline for a document",
                "requires": [
                    {
                        "name": "topic",
                        type: "string",
                        "description": "The topic of the document",
                    }
                ],
            },
            "generate_tags": {
                "description": "Generate tags for the chat content",
                "requires": [],  # No specific parameters required
            },
        }

    def pipes(self):
        """Return available options for the pipe."""
        models = self.valves.MODEL_ID.split(",")
        options = []

        for model in models:
            model_id = model.strip()
            for template_id, template in self.templates.items():
                options.append(
                    {
                        "id": f"{model_id}:{template_id}",
                        "name": f"{model_id} - {template['name']}",
                    }
                )

        return options

    async def detect_action(self, client, user_message, headers=None):
        """
        Use a thinking model to detect the action the user wants to perform.

        Args:
            client: HTTP client
            user_message: The user's message
            headers: HTTP headers for the API request

        Returns:
            dict: The detected action and parameters
        """

        logger.info(f"Detecting action for user message: {user_message}")
        if not headers:
            headers = self._get_api_headers(
                self.valves.THINKING_API_KEY or self.valves.API_KEY
            )

        if "Generate 1-3 broad tags categorizing the main themes of the chat history" in user_message and "JSON format: { \"tags\": [\"tag1\", \"tag2\", \"tag3\"] }" in user_message:
            logger.info("Detected tag generation request from OpenWebUI")
            return {
                "action": "generate_tags",
                "parameters": {},
                "confidence": 1.0
            }
        # Create prompt for action detection with detailed action descriptions
        system_prompt = f"""You are an AI assistant helping with document generation and management tasks.
Your job is to understand what the user is asking for and return a JSON object with the appropriate action.

Available actions:
"""

        # Dynamically generate the list of available actions from self.actions
        for action_name, action_info in self.actions.items():
            system_prompt += (
                f"- {action_name}: {action_info['description']} (requires: "
            )

            requirements = []
            for req in action_info.get("requires", []):
                req_name = req.get("name", "")
                req_type = req.get("type", "string")
                req_desc = req.get("description", "")

                if req_type == "enum":
                    valid_values = ", ".join([f"'{v}'" for v in req.get("values", [])])
                    requirements.append(f"{req_name} (must be one of: {valid_values})")
                elif req_type == "boolean":
                    requirements.append(f"{req_name} (true/false)")
                else:
                    requirements.append(
                        f"{req_name} ({req_desc})" if req_desc else req_name
                    )

            system_prompt += ", ".join(requirements) + ")\n"

        system_prompt += """
Always respond with a JSON array containing objects with these fields:
- action: The action name (one of the options above)
- parameters: Relevant parameters for the action (e.g., topic, section_name, etc.)
- confidence: Your confidence in this interpretation (0.0-1.0)

Example response format:
[
    {
        "action": "write_full_doc",
        "parameters": {
            "topic": "artificial intelligence",
            "template_id": "basic"
        },
        "confidence": 0.9
    }
]"""

        user_prompt = f"What document action should I take based on this request: '{user_message}'?"

        payload = {
            "model": self.valves.THINKING_MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,  # Lower temperature for more consistent output
            "max_tokens": 1000,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # logger.error(f"Payload: {payload}")
                logger.info(f"headers: {headers}")
                response = await client.post(
                    f"{self.valves.THINKING_API_HOST}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )

                response.raise_for_status()
                response_data = response.json()

                # Extract content from assistant's message
                content = (
                    response_data.get("choices", [])[0]
                    .get("message", {})
                    .get("content", "")
                )

                # Clean the content by removing any thinking tags
                logger.error(f"Content: {content}")
                content = self._extract_json_from_response(content)

                # Try to parse the JSON
                try:
                    action_data = json.loads(content)
                    # Validate it's an array with at least one action
                    if isinstance(action_data, list) and len(action_data) > 0:
                        logger.info(f"Action data: {action_data}")
                        return action_data[0]  # Return the highest confidence action
                    else:
                        logger.warning(
                            f"Invalid action data format on attempt {attempt+1}: {content}"
                        )
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON parsing error on attempt {attempt+1}: {e}")
                    if attempt == max_retries - 1:
                        raise
            except Exception as e:
                logger.error(f"Error detecting action on attempt {attempt+1}: {str(e)}")
                if attempt == max_retries - 1:
                    raise

            # If we get here, we need to retry with a clearer prompt
            # Use raw string to avoid any formatting issues
            example_json = r'[{"action": "write_full_doc", "parameters": {"topic": "example topic", "template_id": "basic"}, "confidence": 0.9}]'
            user_prompt = f"""I need a valid JSON response for this request: '{user_message}'
Please return ONLY a JSON array with a single action object like this example:
{example_json}

Remember to choose from one of these actions: write_full_doc, edit_section, add_section, list_sections, summarize_doc, expand_section, rewrite_section, create_summary"""

            payload["messages"] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Wait a bit before retrying
            await asyncio.sleep(1)

        # If all retries failed, return a default action
        return {
            "action": "write_full_doc",
            "parameters": {"topic": user_message, "template_id": "basic"},
            "confidence": 0.5,
        }

    def _extract_json_from_response(self, content):
        """
        Estrae il JSON da una risposta che potrebbe contenere tag <think> o altro testo.
        Se il JSON estratto è una lista con un solo elemento, lo converte in un oggetto JSON.
        Il risultato viene restituito come stringa.
        """
        # Rimuovi eventuali blocchi di "thinking"
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

        # Prova a trovare un array JSON che inizia con [ e termina con ]
        json_match = re.search(r"\[\s*{.*}\s*\]", content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            # Prova a trovare un oggetto JSON che inizia con { e termina con }
            json_match = re.search(r"{.*}", content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                # Se non viene trovato alcun JSON, restituisci il contenuto originale
                json_str = content.replace("```json", "").replace("```", "").strip()
                return json_str

        # Prova a interpretare la stringa estratta come JSON
        try:
            parsed = json.loads(json_str)
            # Se parsed è una lista con un solo elemento, estrai quell'elemento
            if isinstance(parsed, list) and len(parsed) == 1:
                parsed = parsed[0]
            # Restituisci la stringa JSON formattata
            return json.dumps(parsed, indent=2)
        except Exception as e:
            # Se il parsing fallisce, restituisci comunque la stringa pulita
            return json_str

    async def generate_section(self, client, model_id, section, topic, headers):
        """Generate a single section using the API with streaming."""
        prompt = section["prompt"].format(topic=topic)

        # Append clear instructions to force paragraph-only output
        prompt += (
            "\n\nPlease generate only plain text paragraphs without any markdown formatting. "
            "Do not include titles, headings, '#' symbols, or subsections. "
            "Do not include any other text, formatting, or comments. just the content for the section. "
            "The response should serve as the content for an already existing section of a markdown file on the topic: '{}'.".format(
                topic
            )
            + "The response should be in the same language as the topic"
            + "The response should be in the same style as the topic"
            + "The response should be very short and concise"
        )
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,  # Enable streaming
        }

        logger.info(f"Generating section with prompt: {prompt}")

        try:
            # Set up timeout for the request
            timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)

            async with client.stream(
                "POST",
                f"{self.valves.API_HOST}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_message = f"API error ({response.status_code}): {error_text.decode('utf-8')}"
                    logger.error(f"Section generation error: {error_message}")
                    yield {"error": error_message}
                    return

                # Stream the content
                content_buffer = ""
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue

                    chunk_str = chunk.decode("utf-8")
                    # Handle SSE format (data: prefix)
                    for line in chunk_str.split("\n"):
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_piece = delta["content"]
                                        content_buffer += content_piece
                                        # Yield each piece as it arrives
                                        yield {"content_piece": content_piece}
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"JSON parse error: {str(e)} for line: {line}"
                                )

                # Yield the final success status with complete content
                yield {"content": content_buffer, "status": "complete"}

        except httpx.TimeoutException:
            error_message = (
                f"Request timed out after {self.valves.REQUEST_TIMEOUT} seconds"
            )
            logger.error(f"Section generation timeout: {error_message}")
            yield {"error": error_message}
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"Section generation exception: {error_message}")
            yield {"error": error_message}

    async def edit_document_section(
        self, client, model_id, section_name, document, instructions, headers
    ):
        """Edit an existing section of the document."""
        # Find the section in the document
        section_content = document["sections"].get(section_name, "")
        section_info = next(
            (s for s in document["template"]["sections"] if s["name"] == section_name),
            None,
        )

        if not section_info:
            yield {"error": f"Section '{section_name}' not found in the document"}
            return

        # Create a prompt for editing the section
        prompt = f"""Edit the following document section according to these instructions: {instructions}
        
        Section: {section_info['title']}
        
        Current content:
        {section_content}
        
        Please provide the complete edited section:"""

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,
        }

        try:
            # Set up timeout for the request
            timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)

            async with client.stream(
                "POST",
                f"{self.valves.API_HOST}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_message = f"API error ({response.status_code}): {error_text.decode('utf-8')}"
                    logger.error(f"Section editing error: {error_message}")
                    yield {"error": error_message}
                    return

                # Stream the content
                content_buffer = ""
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue

                    chunk_str = chunk.decode("utf-8")
                    # Handle SSE format
                    for line in chunk_str.split("\n"):
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_piece = delta["content"]
                                        content_buffer += content_piece
                                        # Yield each piece as it arrives
                                        yield {"content_piece": content_piece}
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"JSON parse error: {str(e)} for line: {line}"
                                )

                # Yield the final success status with complete content
                yield {"content": content_buffer, "status": "complete"}

        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"Section editing exception: {error_message}")
            yield {"error": error_message}

    async def add_document_section(
        self, client, model_id, document, section_title, instructions, headers
    ):
        """Add a new section to the document."""
        # Create a unique section name
        base_name = re.sub(r"[^a-z0-9]", "", section_title.lower())
        section_name = base_name
        counter = 1

        # Ensure unique section name
        while section_name in document["sections"]:
            section_name = f"{base_name}{counter}"
            counter += 1

        # Create a prompt for the new section
        prompt = f"""Create a new document section with the title "{section_title}" 
        according to these instructions: {instructions}
        
        Document topic: {document.get("topic", "Unknown")}
        
        Please write a complete, well-structured section:"""

        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,
        }

        try:
            # Set up timeout for the request
            timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)

            async with client.stream(
                "POST",
                f"{self.valves.API_HOST}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout,
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    error_message = f"API error ({response.status_code}): {error_text.decode('utf-8')}"
                    logger.error(f"Section addition error: {error_message}")
                    yield {"error": error_message}
                    return

                # Stream the content
                content_buffer = ""
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue

                    chunk_str = chunk.decode("utf-8")
                    # Handle SSE format
                    for line in chunk_str.split("\n"):
                        if line.startswith("data: ") and line != "data: [DONE]":
                            try:
                                data = json.loads(line[6:])
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    if "content" in delta:
                                        content_piece = delta["content"]
                                        content_buffer += content_piece
                                        # Yield each piece as it arrives
                                        yield {
                                            "content_piece": content_piece,
                                            "section_name": section_name,
                                            "section_title": section_title,
                                        }
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"JSON parse error: {str(e)} for line: {line}"
                                )

                # Create the new section metadata
                new_section = {
                    "name": section_name,
                    "title": section_title,
                    "prompt": f"Write content for '{section_title}' about {{topic}}. {instructions}",
                    "children": [],  # Initialize with empty children array
                }

                # Yield the final success status with complete content
                yield {
                    "content": content_buffer,
                    "status": "complete",
                    "section_name": section_name,
                    "section_title": section_title,
                    "section_metadata": new_section,
                }

        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"Section addition exception: {error_message}")
            yield {"error": error_message}

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        print(f"inlet: {__name__}")
        if self.debug:
            print(f"inlet: {__name__} - body:")
            pprint(body)
            print(f"inlet: {__name__} - user:")
            pprint(user)

        logger.info(f"Inlet body: {body}")
        self.user_id = user.get("id")
        self.user_name = user.get("name")
        self.user_email = user.get("email")
        self.chat_id = body.get("metadata").get("chat_id")
        self.message_id = body.get("metadata").get("message_id")
        return body

    async def pipe(
        self, body: dict, __event_emitter__=None, __metadata__=None
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline handler that dispatches to specific action handlers."""
        # Extract data from the request body
        user_message = ""
        model_id = ""
        messages = []
        chat_id = __metadata__.get("chat_id", "default_chat")

        logger.info(f"Pipe body: {body}")

        # Extract user message from body
        if "messages" in body and len(body["messages"]) > 0:
            messages = body["messages"]
            last_user_message = next(
                (m for m in reversed(messages) if m.get("role") == "user"), None
            )
            if last_user_message:
                user_message = last_user_message.get("content", "")

        # Extract model ID from body
        if "model" in body:
            model_id = body["model"]
            # Extract the base model name if it's prefixed with function name
            if "." in model_id:
                model_id = model_id.split(".")[-1]

        # Check if we have an existing document in file storage
        if self.valves.DOCUMENT_STORAGE == "file":
            loaded_document = self.load_document(chat_id)
            if loaded_document:
                self.documents[chat_id] = loaded_document
                logger.info(f"Loaded existing document for chat {chat_id}")

        # Validate API key
        if not self.valves.API_KEY:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Error: API key not configured",
                            "done": True,
                        },
                    }
                )
            yield "Error: API key not configured. Please set the API_KEY valve."
            return

        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openwebui.com/",
            "X-Title": "Open WebUI",
        }

        try:
            # Start the thinking process
            yield "<think>\n"

            # First, detect what action the user wants to perform
            yield "🧠 Analyzing your request...\n"

            # Create a client for API requests
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            timeout = httpx.Timeout(self.valves.TIMEOUT, connect=5.0)

            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                # Detect the action
                action_result = await self.detect_action(client, user_message, headers)

                if "error" in action_result:
                    yield f"Error detecting action: {action_result['error']}\n"
                    yield f"Falling back to default document generation...\n"
                    action = "write_full_doc"
                    params = {
                        "topic": user_message,
                        "template_id": self.valves.TEMPLATE_NAME,
                    }
                else:
                    action = action_result["action"]
                    # Extract parameters correctly from the nested structure
                    params = action_result.get("parameters", {})
                    confidence = action_result.get("confidence", 0.0)

                    yield f"Detected action: {self.actions[action]['description']} {action} (confidence: {confidence:.2f})\n"
                    yield f"Parameters: {json.dumps(params, indent=2)}\n\n"

                # Dispatch to appropriate handler based on action
                if action == "write_full_doc":
                    async for chunk in self.handle_write_full_doc(
                        client,
                        chat_id,
                        user_message,
                        model_id,
                        params,
                        headers,
                        __event_emitter__,
                    ):
                        yield chunk
                elif action == "edit_section":
                    async for chunk in self.handle_edit_section(
                        client, chat_id, params, headers, __event_emitter__
                    ):
                        yield chunk
                elif action == "add_section":
                    async for chunk in self.handle_add_section(
                        client,
                        chat_id,
                        user_message,
                        params,
                        headers,
                        __event_emitter__,
                    ):
                        yield chunk
                elif action == "list_sections":
                    async for chunk in self.handle_list_sections(chat_id):
                        yield chunk
                elif action == "summarize_doc":
                    async for chunk in self.handle_summarize_doc(
                        client, chat_id, params, headers, __event_emitter__
                    ):
                        yield chunk
                elif action == "expand_section":
                    async for chunk in self.handle_expand_section(
                        client, chat_id, params, headers, __event_emitter__
                    ):
                        yield chunk
                elif action == "rewrite_section":
                    async for chunk in self.handle_rewrite_section(
                        client, chat_id, params, headers, __event_emitter__
                    ):
                        yield chunk
                elif action == "create_summary":
                    async for chunk in self.handle_create_summary(
                        client, params, headers, __event_emitter__
                    ):
                        yield chunk
                elif action == "generate_tags":
                    # For tag generation, we don't use the thinking markers
                    result = await self.handle_generate_tags(
                        client, user_message, headers
                    )
                    # Return the result directly without thinking markers
                    yield result
                    return
                else:
                    # Default case - unrecognized action
                    yield f"Action '{action}' not implemented yet. Falling back to document listing.\n"
                    yield "</think>\n\n"

                    if chat_id in self.documents:
                        document = self.documents[chat_id]

                        yield f"# Current Document: {document['topic']}\n\n"
                        yield f"**Template:** {document['template']['name']}\n"
                        yield f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['created_at']))}\n\n"

                        yield "## Available Actions:\n\n"
                        yield "- Generate a new document: *Write a document about [topic]*\n"
                        yield "- Edit a section: *Edit the Introduction section to include more details*\n"
                        yield "- Add a section: *Add a section called 'Future Work' after the Conclusion*\n"
                        yield "- List sections: *Show me the sections in my document*\n"
                    else:
                        yield "# Document Generator\n\n"
                        yield "No document has been created yet. You can:\n\n"
                        yield "- Generate a new document: *Write a document about [topic]*\n"

            # Send completion status
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {"description": "Operation completed", "done": True},
                    }
                )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Document operation error: {error_msg}")

            # Send error status
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": error_msg, "done": True}}
                )

            # If we're in the middle of thinking, close the tag
            yield "</think>\n\n"

    async def handle_write_full_doc(
        self,
        client,
        chat_id,
        user_message,
        model_id,
        params,
        headers,
        __event_emitter__=None,
    ):
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

        # Send status update
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating document on '{topic}'",
                        "done": False,
                    },
                }
            )

        yield f"Generating full document on '{topic}' using template: {template['name']}\n\n"

        # Generate the document with nested sections
        document = {
            "topic": topic,
            "template": template,
            "model_id": llm_model_id,
            "sections": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # Generate all sections including nested ones
        async for chunk in self.generate_document(
            client, llm_model_id, template, topic, headers
        ):
            yield chunk

        # Get updated document from self if available
        if hasattr(self, "current_document") and self.current_document:
            document = self.current_document
            self.current_document = None  # Clear it after use

        # Store the document
        self.documents[chat_id] = document
        self.save_document(chat_id, document)  # Save to file if enabled

        # End the thinking process
        yield "\nCompiling final document...\n"
        yield "</think>\n\n"

        # Return the complete formatted document
        yield self.format_document(document)

    async def handle_edit_section(
        self, client, chat_id, params, headers, __event_emitter__=None
    ):
        """Handle editing a specific section."""
        section_name = params.get("section_name", "")
        instructions = params.get("instructions", "Improve this section")

        # Check if we have a document
        if chat_id not in self.documents:
            yield "No document found to edit. Please generate a document first."
            yield "</think>\n\n"
            return

        document = self.documents[chat_id]
        llm_model_id = document["model_id"]

        # Find section using the helper function
        section, section_path = self.find_section_by_name(document, section_name)

        if not section or section["name"] not in document["sections"]:
            yield f"Section '{section_name}' not found in the document.\n"
            yield "Available sections:\n"
            yield self.list_document_sections(document)
            yield "</think>\n\n"

            # Return a helpful message to the user
            yield f"# Section Not Found\n\n"
            yield f"I couldn't find a section named '{section_name}' in your document. Here are the available sections:\n\n"
            yield self.list_document_sections(document)
            return

        # Get section information
        section_title = section["title"]
        section_name = section["name"]

        yield f"Editing section: {section_title} ({section_name}) with instructions: {instructions}\n\n"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Editing section: {section_title}",
                        "done": False,
                    },
                }
            )

        # Stream the section editing
        section_content = ""
        async for chunk in self.edit_document_section(
            client, llm_model_id, section_name, document, instructions, headers
        ):
            if "error" in chunk:
                yield f"Error editing section: {chunk['error']}\n"
                break
            elif "content_piece" in chunk:
                section_content += chunk["content_piece"]
                yield f"{chunk['content_piece']}"
            elif "status" in chunk and chunk["status"] == "complete":
                document["sections"][section_name] = chunk.get(
                    "content", section_content
                )
                document["updated_at"] = time.time()
                self.save_document(chat_id, document)  # Save to file if enabled
                yield f"\nCompleted editing section: {section_title}\n"

        # End thinking and show the entire document
        yield "</think>\n\n"

        # Format and display the document, highlighting the edited section
        yield self.format_document(document, highlight_section=section_name)

    async def determine_section_position(
        self, client, document, section_title, position_info, headers
    ):
        """
        Determine where to place a new section in the document based on user instructions.

        Returns positioning information including parent section and reference section.
        """
        # Create a simplified representation of the document structure for the thinking model
        doc_structure = []

        def extract_section_structure(sections, parent=None, level=0):
            result = []
            for section in sections:
                section_info = {
                    "name": section["name"],
                    "title": section["title"],
                    "level": level,
                    "parent": parent,
                }
                result.append(section_info)

                if "children" in section and section["children"]:
                    child_results = extract_section_structure(
                        section["children"], parent=section["name"], level=level + 1
                    )
                    result.extend(child_results)
            return result

        doc_structure = extract_section_structure(document["template"]["sections"])

        logger.info(f"Document structure: {str(doc_structure)}")

        # Create prompt for the thinking model
        system_prompt = """You are an AI assistant helping with document structure organization.
Your task is to determine where to place a new section in an existing document based on the user's request.
Analyze the document structure and the positioning information to identify the optimal placement.

Return a JSON object with these fields:
- parent_section: The parent section name (or null if it's a top-level section)
- reference_section: The name of the section to place this new section before or after
- position: Either "before" or "after" to indicate relative positioning
- explanation: A brief explanation of why this position is appropriate

Example response:
{
  "parent_section": null,
  "reference_section": "introduction", 
  "position": "after",
  "explanation": "User's positioning request was to add this section after the introduction."
}"""

        user_prompt = f"""Current document structure:
{json.dumps(doc_structure, indent=2)}

New section title: "{section_title}"

User's positioning request: "{position_info}"

Where should this new section be placed in the document structure? Return the position as a JSON object."""

        payload = {
            "model": self.valves.THINKING_MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
        }

        try:
            response = await client.post(
                f"{self.valves.THINKING_API_HOST}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            response_data = response.json()

            # Extract content from assistant's message
            content = (
                response_data.get("choices", [])[0]
                .get("message", {})
                .get("content", "")
            )

            # Clean the content and extract JSON
            content = self._extract_json_from_response(content)

            # Parse the positioning information
            position_data = json.loads(content)

            # Log the raw and parsed position data for debugging
            logger.info(f"Raw position data: {content}")
            logger.info(f"Parsed position data type: {type(position_data)}")
            logger.info(f"Parsed position data: {position_data}")

            # Handle both list and dictionary responses
            if isinstance(position_data, list) and len(position_data) > 0:
                # If it's a list, take the first item
                logger.info("Position data is a list, extracting first item")
                position_data = position_data[0]

            # Validate the position data has the required fields
            required_fields = ["reference_section", "position"]
            for field in required_fields:
                if field not in position_data:
                    logger.warning(f"Missing required field '{field}' in position data")
                    position_data[field] = (
                        "end" if field == "reference_section" else "after"
                    )

            return position_data

        except Exception as e:
            logger.error(f"Error determining section position: {str(e)}")
            # Return a default position at the end of the document
            return {
                "parent_section": None,
                "reference_section": "end",
                "position": "after",
                "explanation": f"Default position due to error: {str(e)}",
            }

    def insert_section_into_template(self, document, section_metadata, position_data):
        """
        Insert a new section into the document template at the specified position.

        Args:
            document: The document to modify
            section_metadata: The metadata for the new section
            position_data: Positioning information from determine_section_position

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Log the input parameters for debugging
            logger.info(
                f"Inserting section: {section_metadata['name']} ({section_metadata['title']})"
            )
            logger.info(f"Position data: {position_data}")

            # Get positioning information
            reference_section = position_data.get("reference_section")
            position = position_data.get("position", "after")
            parent_section = position_data.get("parent_section")

            # If reference section is "end", append to the end of the document or parent
            if reference_section == "end" or not reference_section:
                logger.info(f"Adding section to the end of document")
                document["template"]["sections"].append(section_metadata)
                return True

            # Function to recursively find and insert the section
            def find_and_insert(sections, parent_name=None):
                for i, section in enumerate(sections):
                    # Check if this is the reference section
                    if section["name"] == reference_section:
                        # Found the reference section
                        logger.info(
                            f"Found reference section: {section['name']} at level {parent_name or 'root'}"
                        )

                        # If we need a specific parent but this doesn't match, continue searching
                        if parent_section and parent_name != parent_section:
                            continue

                        # Insert before or after based on position
                        if position == "before":
                            logger.info(f"Inserting before index {i}")
                            sections.insert(i, section_metadata)
                        else:  # after
                            logger.info(f"Inserting after index {i}")
                            sections.insert(i + 1, section_metadata)
                        return True

                    # Check children if this section has them
                    if "children" in section and section["children"]:
                        if find_and_insert(section["children"], section["name"]):
                            return True

                # If we get here and haven't found the section but were asked to add at parent level
                if parent_section == parent_name:
                    logger.info(
                        f"Adding to parent {parent_name} as reference section not found"
                    )
                    sections.append(section_metadata)
                    return True

                return False

            # Try to insert at the specified position
            success = find_and_insert(document["template"]["sections"])

            if not success:
                logger.warning(
                    f"Could not find reference section {reference_section}, adding to end"
                )
                document["template"]["sections"].append(section_metadata)

            return True

        except Exception as e:
            logger.error(f"Error inserting section: {str(e)}")
            # Add to the end as a fallback
            try:
                document["template"]["sections"].append(section_metadata)
                return True
            except:
                return False

    async def handle_add_section(
        self, client, chat_id, user_message, params, headers, __event_emitter__=None
    ):
        """Handle adding a new section to the document."""
        section_title = params.get("section_title", "New Section")
        instructions = params.get("instructions", "Add content for this section")
        # position_info = params.get("position", "")  # Extract positioning information

        # Check if we have a document
        if chat_id not in self.documents:
            yield "No document found to modify. Please generate a document first."
            yield "</think>\n\n"
            return

        document = self.documents[chat_id]
        llm_model_id = document["model_id"]

        yield f"Adding new section: {section_title} with instructions: {instructions}\n\n"

        # Determine section position based on user instructions
        yield f"Analyzing where to place the section based on: {user_message}\n"
        try:
            position_data = await self.determine_section_position(
                # client, document, section_title, position_info or user_message, headers
                client,
                document,
                section_title,
                user_message,
                headers,
            )

            yield f"Position determined: {json.dumps(position_data, indent=2)}\n\n"
        except Exception as e:
            logger.error(f"Error in position determination: {str(e)}")
            position_data = {
                "parent_section": None,
                "reference_section": "end",
                "position": "after",
                "explanation": "Default position due to error",
            }
            yield f"Error determining position: {str(e)}. Will add to the end of document.\n\n"

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Adding section: {section_title}",
                        "done": False,
                    },
                }
            )

        # Stream the section creation
        section_content = ""
        section_name = None
        section_metadata = None

        async for chunk in self.add_document_section(
            client, llm_model_id, document, section_title, instructions, headers
        ):
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
                    # Add content to the sections object
                    document["sections"][section_name] = section_content

                    # Add the section to the template at the right position
                    success = self.insert_section_into_template(
                        document, section_metadata, position_data
                    )

                    if success:
                        document["updated_at"] = time.time()
                        self.save_document(chat_id, document)  # Save to file if enabled
                        yield f"\nCompleted adding section: {section_title}\n"
                        ref_section = position_data.get("reference_section", "the end")
                        pos = position_data.get("position", "after")
                        yield f"Section placed {pos} {ref_section}\n"
                    else:
                        yield f"\nError positioning the section. Added at the end instead.\n"
                        document["template"]["sections"].append(section_metadata)
                        document["updated_at"] = time.time()
                        self.save_document(chat_id, document)

        # End thinking and show the entire document
        yield "</think>\n\n"

        # Format and display the document, highlighting the edited section
        yield self.format_document(document, highlight_section=section_name)

    async def handle_list_sections(self, chat_id):
        """Handle listing all sections in the document."""
        # Check if we have a document
        if chat_id not in self.documents:
            yield "No document found. Please generate a document first."
            yield "</think>\n\n"
            return

        document = self.documents[chat_id]

        yield f"Listing sections for document on topic: {document['topic']}\n\n"

        # End thinking
        yield "</think>\n\n"

        # Use the hierarchical section listing
        yield self.list_document_sections(document)

    async def handle_summarize_doc(
        self, client, chat_id, params, headers, __event_emitter__=None
    ):
        """Handle summarizing the existing document."""
        # Placeholder - implement the actual logic
        yield "Summarize document action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Document Summary\n\nThis feature is coming soon!"

    async def handle_expand_section(
        self, client, chat_id, params, headers, __event_emitter__=None
    ):
        """Handle expanding a specific section with more details."""
        # Placeholder - implement the actual logic
        yield "Expand section action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Section Expansion\n\nThis feature is coming soon!"

    async def handle_rewrite_section(
        self, client, chat_id, params, headers, __event_emitter__=None
    ):
        """Handle rewriting a section in a different style."""
        # Placeholder - implement the actual logic
        yield "Rewrite section action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Section Rewriting\n\nThis feature is coming soon!"

    async def handle_create_summary(
        self, client, params, headers, __event_emitter__=None
    ):
        """Handle creating an outline for a document."""
        # Placeholder - implement the actual logic
        yield "Create outline action not fully implemented yet."
        yield "</think>\n\n"
        yield "# Document Outline\n\nThis feature is coming soon!"

    def format_document(self, document, highlight_section=None, include_metadata=True):
        """
        Format a document as markdown for display to the user with support for nested sections.

        Args:
            document: The document object to format
            highlight_section: Optional section name to highlight as updated
            include_metadata: Whether to include document metadata

        Returns:
            Formatted markdown string
        """
        output = []

        # Show the title if it exists
        if "topic" in document:
            output.append(f"# Generated Document: {document['topic']}\n")
        else:
            output.append(f"# Generated Document\n")

        # Helper function to recursively format sections
        def format_section(section, level=2):
            section_key = section["name"]
            heading_prefix = "#" * level

            if section_key in document["sections"]:
                # Highlight the edited section if specified
                if section_key == highlight_section:
                    output.append(f"{heading_prefix} {section['title']} (Updated)\n")
                else:
                    output.append(f"{heading_prefix} {section['title']}\n")

                output.append(f"{document['sections'][section_key]}\n\n")

                # Process child sections if any
                if "children" in section and section["children"]:
                    for child in section["children"]:
                        format_section(child, level + 1)

        # Process all top-level sections
        for section in document["template"]["sections"]:
            format_section(section)

        # Add metadata if requested
        if include_metadata:
            output.append("---\n")
            if "created_at" in document:
                output.append(
                    f"Document created: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['created_at']))}\n"
                )
            if "updated_at" in document:
                output.append(
                    f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['updated_at']))}\n"
                )
            if "model_id" in document:
                output.append(f"Generated using model: {document['model_id']}\n")
            if "template" in document and "name" in document["template"]:
                output.append(f"Template: {document['template']['name']}\n")

        return "\n".join(output)

    def _get_document_path(self, chat_id):
        """Get the file path for a document based on chat ID."""
        # Ensure the directory exists
        os.makedirs("/app/backend/data/doc_generator_pipe", exist_ok=True)
        return f"/app/backend/data/doc_generator_pipe/{chat_id}.json"

    def save_document(self, chat_id, document):
        """Save a document to file storage."""
        if self.valves.DOCUMENT_STORAGE != "file":
            # Only save if file storage is enabled
            return

        try:
            file_path = self._get_document_path(chat_id)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(document, f, ensure_ascii=False, indent=2)
            logger.info(f"Document saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving document: {str(e)}")

    def load_document(self, chat_id):
        """Load a document from file storage."""
        if self.valves.DOCUMENT_STORAGE != "file":
            # Return None if not using file storage
            return None

        try:
            file_path = self._get_document_path(chat_id)
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    document = json.load(f)
                logger.info(f"Document loaded from {file_path}")
                return document
            else:
                logger.info(f"No document found at {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            return None

    async def generate_document(self, client, llm_model_id, template, topic, headers):
        """Generate a complete document with nested sections."""
        document = {
            "topic": topic,
            "template": template,
            "model_id": llm_model_id,
            "sections": {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }

        # Helper function to recursively generate sections
        async def generate_section_recursive(section, parent_path=""):
            section_name = section["title"]
            section_key = section["name"]

            yield f"Generating section: {section_name}...\n"

            # Initialize empty content for this section
            section_content = ""
            has_error = False

            # Stream the section generation
            async for chunk in self.generate_section(
                client, llm_model_id, section, topic, headers
            ):
                if "error" in chunk:
                    yield f"Failed to generate section '{section_name}'. Moving to next section...\n"
                    has_error = True
                    break
                elif "content_piece" in chunk:
                    # Accumulate content
                    section_content += chunk["content_piece"]
                    # Stream content piece to user
                    yield chunk["content_piece"]
                elif "status" in chunk and chunk["status"] == "complete":
                    # Section completed
                    document["sections"][section_key] = chunk.get(
                        "content", section_content
                    )
                    yield f"\nCompleted section: {section_name}\n"

            # If no error occurred and we didn't get a completion status
            if (
                not has_error
                and section_key not in document["sections"]
                and section_content
            ):
                document["sections"][section_key] = section_content
                yield f"\nCompleted section: {section_name}\n"

            # Generate child sections if any
            if "children" in section and section["children"]:
                for child in section["children"]:
                    async for chunk in generate_section_recursive(
                        child, f"{parent_path}/{section_key}"
                    ):
                        yield chunk

        # Generate all top-level sections and their children
        for section in template["sections"]:
            async for chunk in generate_section_recursive(section):
                yield chunk

        # Store document for reference but don't return it
        self.current_document = document
        # No return statement - this fixes the async generator error

    def find_section_by_name(self, document, section_name):
        """Find a section by name or title in a nested document structure."""
        section_name_lower = section_name.lower()

        def search_section(sections, parent_path=""):
            for section in sections:
                current_name = section["name"]
                current_path = (
                    f"{parent_path}/{current_name}" if parent_path else current_name
                )

                # Check if this section matches
                if (
                    current_name.lower() == section_name_lower
                    or section["title"].lower() == section_name_lower
                ):
                    return section, current_path

                # Check children if any
                if "children" in section and section["children"]:
                    result = search_section(section["children"], current_path)
                    if result:
                        return result

            return None, None

        section, path = search_section(document["template"]["sections"])
        return section, path

    def list_document_sections(self, document):
        """Generate a markdown listing of all sections in the document with their hierarchy."""
        output = [f"# Document Sections for: {document.get('topic', 'Untitled')}\n"]

        def format_section_listing(sections, indent=0):
            for section in sections:
                section_name = section["name"]
                section_title = section["title"]
                prefix = "  " * indent + "- "

                if section_name in document["sections"]:
                    word_count = len(document["sections"][section_name].split())
                    output.append(f"{prefix}**{section_title}** ({word_count} words)\n")
                else:
                    output.append(f"{prefix}{section_title} (not generated)\n")

                # Process children if any
                if "children" in section and section["children"]:
                    format_section_listing(section["children"], indent + 1)

        format_section_listing(document["template"]["sections"])

        # Add document metadata
        output.append(f"\n**Topic:** {document.get('topic', 'Unspecified')}\n")
        output.append(f"**Template:** {document['template']['name']}\n")
        if "created_at" in document:
            output.append(
                f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['created_at']))}\n"
            )
        if "updated_at" in document:
            output.append(
                f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['updated_at']))}\n"
            )

        return "\n".join(output)

    def load_git_templates(self):
        """Load additional templates from specified Git repositories."""
        logger.info("Loading templates from Git repositories")

        # Process the comma-separated list of repositories
        if (
            not self.valves.GIT_TEMPLATE_REPOS
            or self.valves.GIT_TEMPLATE_REPOS.strip() == ""
        ):
            logger.info("No Git repositories specified for templates")
            return

        # Split the comma-separated string and clean each URL
        repo_urls = [
            url.strip()
            for url in self.valves.GIT_TEMPLATE_REPOS.split(",")
            if url.strip()
        ]
        logger.info(f"Found {len(repo_urls)} repositories to process")

        for repo_url in repo_urls:
            try:
                logger.info(f"Processing Git repository: {repo_url}")
                repo_name = self._get_repo_name(repo_url)
                repo_dir = f"/app/backend/data/doc_generator_pipe/git/{repo_name}"

                # Clone or update the repository
                self._clone_or_update_repo(repo_url, repo_dir)

                # Copy template files
                self._copy_template_files(repo_dir)

            except Exception as e:
                logger.error(f"Error processing Git repository {repo_url}: {str(e)}")

        # Load templates from local template directory
        self._load_local_templates()

    def _get_repo_name(self, repo_url):
        """Extract repository name from URL."""
        # Extract the last part of the URL and remove .git extension if present
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]

        # Sanitize the name to ensure it's a valid directory name
        repo_name = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)
        return repo_name

    def _clone_or_update_repo(self, repo_url, repo_dir):
        """Clone a Git repository or pull latest changes if it exists."""
        try:
            if os.path.exists(repo_dir):
                # Repository already exists, pull latest changes
                logger.info(f"Updating existing repository: {repo_dir}")
                result = subprocess.run(
                    ["git", "-C", repo_dir, "pull"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Git pull result: {result.stdout}")
            else:
                # Clone the repository
                logger.info(f"Cloning repository: {repo_url} to {repo_dir}")
                result = subprocess.run(
                    ["git", "clone", repo_url, repo_dir],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                logger.info(f"Git clone result: {result.stdout}")

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Error in Git operation: {str(e)}")
            return False

    def _copy_template_files(self, repo_dir):
        """Copy template JSON files from the repository to the templates directory."""
        templates_path = os.path.join(repo_dir, "templates")

        if not os.path.exists(templates_path):
            logger.warning(f"No templates directory found in {repo_dir}")
            return

        # Find all JSON files in the templates directory
        template_files = glob.glob(os.path.join(templates_path, "*.json"))
        logger.info(f"Found {len(template_files)} template files in {templates_path}")

        for template_file in template_files:
            try:
                # Get the template file name
                template_filename = os.path.basename(template_file)

                # Create a timestamped version to avoid conflicts
                timestamp = int(time.time())
                timestamped_filename = (
                    f"{os.path.splitext(template_filename)[0]}_{timestamp}.json"
                )

                # Copy to the templates directory
                destination = f"/app/backend/data/doc_generator_pipe/templates/{timestamped_filename}"
                shutil.copy2(template_file, destination)
                logger.info(f"Copied template: {template_file} -> {destination}")

            except Exception as e:
                logger.error(f"Error copying template file {template_file}: {str(e)}")

    def _load_local_templates(self):
        """Load templates from the local templates directory."""
        template_files = glob.glob(
            "/app/backend/data/doc_generator_pipe/templates/*.json"
        )
        logger.info(f"Loading {len(template_files)} templates from local directory")

        for template_file in template_files:
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    template_data = json.load(f)

                if (
                    not isinstance(template_data, dict)
                    or "name" not in template_data
                    or "sections" not in template_data
                ):
                    logger.warning(
                        f"Invalid template format in {template_file}. Skipping."
                    )
                    continue

                # Create a unique template ID from filename
                template_id = os.path.splitext(os.path.basename(template_file))[0]

                # Add template to the templates dictionary
                self.templates[template_id] = template_data
                logger.info(
                    f"Added template: {template_id} ({template_data.get('name')})"
                )

            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in template file {template_file}")
            except Exception as e:
                logger.error(f"Error loading template from {template_file}: {str(e)}")

    async def handle_generate_tags(self, client, message, headers):
        """Handle generating tags for the chat content."""
        logger.info("Generating tags for chat content")

        # Extract the actual request details from the message
        # This is the prompt sent by OpenWebUI
        tag_request = message

        # Prepare request for the thinking model
        system_prompt = """You are an expert at categorizing conversations and content.
Your task is to generate appropriate tags that categorize the main themes discussed.
Follow the guidelines exactly and provide only the requested JSON output, example  output:
{ "tags": ["tag1", "tag2", "tag3"] }"""

        payload = {
            "model": self.valves.THINKING_MODEL_ID,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": tag_request},
            ],
            "temperature": 0.3,
        }

        try:
            response = await client.post(
                f"{self.valves.THINKING_API_HOST}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            response.raise_for_status()
            response_data = response.json()

            # Extract content from assistant's message
            content = (
                response_data.get("choices", [])[0]
                .get("message", {})
                .get("content", "")
            )

            # Clean the content to ensure it's proper JSON
            content = self._extract_json_from_response(content)

            logger.info(f"Generated tags: {content}")

            # Return the tags directly without thinking markers
            return content

        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            # Return a basic tag set as fallback
            return '{"tags": ["General"]}'
