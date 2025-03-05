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

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Pipe:
    class Valves(BaseModel):
        TEMPLATE_NAME: str = Field(
            default="basic",
            description="The template to use for document generation"
        )
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

    def __init__(self):
        self.valves = self.Valves()
        # Store for documents in memory
        self.documents = {}
        # Embedded templates - will be moved to external storage in future
        self.templates = {
            "basic": {
                "name": "Basic Document",
                "description": "A simple document with introduction, body, and conclusion",
                "sections": [
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an introduction about {topic}. Provide context and background."
                    },
                    {
                        "name": "body",
                        "title": "Main Content",
                        "prompt": "Provide detailed information about {topic}. Include key facts, analysis, and insights."
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion summarizing the key points about {topic}."
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
                        "prompt": "Write a concise abstract for a research paper about {topic}."
                    },
                    {
                        "name": "introduction",
                        "title": "Introduction",
                        "prompt": "Write an academic introduction for research on {topic}. Include research questions."
                    },
                    {
                        "name": "literature",
                        "title": "Literature Review",
                        "prompt": "Provide a literature review for research on {topic}. Reference key studies and findings."
                    },
                    {
                        "name": "methodology",
                        "title": "Methodology",
                        "prompt": "Describe a research methodology for studying {topic}."
                    },
                    {
                        "name": "results",
                        "title": "Results and Discussion",
                        "prompt": "Present hypothetical results and discussion for research on {topic}."
                    },
                    {
                        "name": "conclusion",
                        "title": "Conclusion",
                        "prompt": "Write a conclusion for academic research on {topic}. Include limitations and future directions."
                    },
                    {
                        "name": "references",
                        "title": "References",
                        "prompt": "Generate a sample reference list for research on {topic}. Use proper academic citation format."
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
                        "prompt": "Generate a sample section for a document on {topic}."
                    }
                ]
            }
        }
        
        # Define available actions
        self.actions = {
            "write_full_doc": {
                "description": "Generate a complete document",
                "requires": [{"name": "topic", type: "string", "description": "The topic of the document"}, {"name": "template_id", type: "enum", "values": list(self.templates.keys()), "description": "The template to use for document generation"}]
            },
            "edit_section": {
                "description": "Edit or regenerate a specific section",
                "requires": [{"name": "section_name", type: "string", "description": "The name of the section to edit"}, {"name": "instructions", type: "string", "description": "The instructions for the section"}]
            },
            "add_section": {
                "description": "Add a new section to the document",
                "requires": [{"name": "section_title", type: "string"}, {"name": "instructions", type: "string"}]
            },
            "summarize_doc": {
                "description": "Summarize the existing document",
                "requires": [{"name": "chat_id", type: "string", "description": "The chat ID of the document"}]
            },
            "expand_section": {
                "description": "Expand a specific section with more details",
                "requires": [{"name": "section_name", type: "string", "description": "The name of the section to expand"}, {"name": "instructions", type: "string", "description": "The instructions for the section"}]
            },
            "rewrite_section": {
                "description": "Rewrite a section in a different style",
                "requires": [{"name": "section_name", type: "string", "description": "The name of the section to rewrite"}, {"name": "style", type: "string", "description": "The style to rewrite the section in"}]
            },
            "list_sections": {
                "description": "List all sections in the current document",
                "requires": [{"name": "chat_id", type: "string", "description": "The chat ID of the document"}]
            },
            "create_summary": {
                "description": "Create an outline for a document",
                "requires": [{"name": "topic", type: "string", "description": "The topic of the document"}]
            }
        }

    def pipes(self):
        """Return available options for the pipe."""
        models = self.valves.MODEL_ID.split(",")
        options = []
        
        for model in models:
            model_id = model.strip()
            for template_id, template in self.templates.items():
                options.append({
                    "id": f"{model_id}:{template_id}",
                    "name": f"{model_id} - {template['name']}",
                })
        
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
        if not headers:
            headers = self._get_api_headers(self.valves.THINKING_API_KEY or self.valves.API_KEY)
            
        # Create prompt for action detection with detailed action descriptions
        system_prompt = f"""You are an AI assistant helping with document generation and management tasks.
Your job is to understand what the user is asking for and return a JSON object with the appropriate action.

Available actions:
"""

        # Dynamically generate the list of available actions from self.actions
        for action_name, action_info in self.actions.items():
            system_prompt += f"- {action_name}: {action_info['description']} (requires: "
            
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
                    requirements.append(f"{req_name} ({req_desc})" if req_desc else req_name)
            
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
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3,  # Lower temperature for more consistent output
            "max_tokens": 1000
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
                    timeout=30
                )

                response.raise_for_status()
                response_data = response.json()
                
                # Extract content from assistant's message
                content = response_data.get("choices", [])[0].get("message", {}).get("content", "")
                
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
                        logger.warning(f"Invalid action data format on attempt {attempt+1}: {content}")
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
                {"role": "user", "content": user_prompt}
            ]
            
            # Wait a bit before retrying
            await asyncio.sleep(1)
        
        # If all retries failed, return a default action
        return {
            "action": "write_full_doc",
            "parameters": {
                "topic": user_message,
                "template_id": "basic"
            },
            "confidence": 0.5
        }
    
    def _extract_json_from_response(self, content):
        """
        Extract JSON from a response that might contain <think> tags or other text
        """
        # First, try to remove any thinking blocks
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        
        # Look for JSON arrays starting with [ and ending with ]
        json_match = re.search(r'\[\s*{.*}\s*\]', content, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        # Try to find JSON objects starting with { and ending with }
        json_match = re.search(r'{.*}', content, re.DOTALL)
        if json_match:
            # Wrap it in an array
            return f"[{json_match.group(0)}]"
        
        # If all else fails, return the original content
        # Cleaned up to remove markdown backticks
        return content.replace('```json', '').replace('```', '').strip()

    async def generate_section(self, client, model_id, section, topic, headers):
        """Generate a single section using the API with streaming."""
        prompt = section["prompt"].format(topic=topic)
        
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.valves.TEMPERATURE,
            "max_tokens": self.valves.MAX_TOKENS,
            "stream": True,  # Enable streaming
        }
        
        try:
            # Set up timeout for the request
            timeout = httpx.Timeout(self.valves.REQUEST_TIMEOUT, connect=5.0)
            
            async with client.stream(
                "POST",
                f"{self.valves.API_HOST}/chat/completions",
                json=payload,
                headers=headers,
                timeout=timeout
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
                    
                    chunk_str = chunk.decode('utf-8')
                    # Handle SSE format (data: prefix)
                    for line in chunk_str.split('\n'):
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content_piece = delta['content']
                                        content_buffer += content_piece
                                        # Yield each piece as it arrives
                                        yield {"content_piece": content_piece}
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parse error: {str(e)} for line: {line}")
                
                # Yield the final success status with complete content
                yield {"content": content_buffer, "status": "complete"}
                
        except httpx.TimeoutException:
            error_message = f"Request timed out after {self.valves.REQUEST_TIMEOUT} seconds"
            logger.error(f"Section generation timeout: {error_message}")
            yield {"error": error_message}
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"Section generation exception: {error_message}")
            yield {"error": error_message}

    async def edit_document_section(self, client, model_id, section_name, document, instructions, headers):
        """Edit an existing section of the document."""
        # Find the section in the document
        section_content = document["sections"].get(section_name, "")
        section_info = next((s for s in document["template"]["sections"] if s["name"] == section_name), None)
        
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
                timeout=timeout
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
                    
                    chunk_str = chunk.decode('utf-8')
                    # Handle SSE format
                    for line in chunk_str.split('\n'):
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content_piece = delta['content']
                                        content_buffer += content_piece
                                        # Yield each piece as it arrives
                                        yield {"content_piece": content_piece}
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parse error: {str(e)} for line: {line}")
                
                # Yield the final success status with complete content
                yield {"content": content_buffer, "status": "complete"}
                
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"Section editing exception: {error_message}")
            yield {"error": error_message}

    async def add_document_section(self, client, model_id, document, section_title, instructions, headers):
        """Add a new section to the document."""
        # Create a unique section name
        base_name = re.sub(r'[^a-z0-9]', '', section_title.lower())
        section_name = base_name
        counter = 1
        
        # Ensure unique section name
        while section_name in [s["name"] for s in document["template"]["sections"]]:
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
                timeout=timeout
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
                    
                    chunk_str = chunk.decode('utf-8')
                    # Handle SSE format
                    for line in chunk_str.split('\n'):
                        if line.startswith('data: ') and line != 'data: [DONE]':
                            try:
                                data = json.loads(line[6:])
                                if 'choices' in data and len(data['choices']) > 0:
                                    delta = data['choices'][0].get('delta', {})
                                    if 'content' in delta:
                                        content_piece = delta['content']
                                        content_buffer += content_piece
                                        # Yield each piece as it arrives
                                        yield {"content_piece": content_piece, "section_name": section_name, "section_title": section_title}
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parse error: {str(e)} for line: {line}")
                
                # Create the new section metadata
                new_section = {
                    "name": section_name,
                    "title": section_title,
                    "prompt": f"Write content for '{section_title}' about {{topic}}. {instructions}"
                }
                
                # Yield the final success status with complete content
                yield {
                    "content": content_buffer, 
                    "status": "complete", 
                    "section_name": section_name,
                    "section_title": section_title,
                    "section_metadata": new_section
                }
                
        except Exception as e:
            error_message = f"Exception: {str(e)}"
            logger.error(f"Section addition exception: {error_message}")
            yield {"error": error_message}

    async def pipe(self, body: dict, __event_emitter__=None) -> Union[str, Generator, Iterator]:
        """
        Smart document generator that detects and performs different document operations.
        
        Args:
            body: Contains user message, model ID, and messages history
            __event_emitter__: Optional function to emit status events
            
        Returns:
            Generated or modified document content
        """
        # Extract data from the request body
        logger.debug(f"Document request received: {body}")
        
        # Extract required information from body
        user_message = ""
        model_id = ""
        messages = []
        chat_id = body.get("chat_id", "default")
        
        # Extract user message from body
        if "messages" in body and len(body["messages"]) > 0:
            messages = body["messages"]
            last_user_message = next((m for m in reversed(messages) if m.get("role") == "user"), None)
            if last_user_message:
                user_message = last_user_message.get("content", "")
        
        # Extract model ID from body
        if "model" in body:
            model_id = body["model"]
            # Extract the base model name if it's prefixed with function name
            if "." in model_id:
                # Get the part after the last dot
                model_id = model_id.split(".")[-1]
        
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
                    params = {"topic": user_message, "template_id": self.valves.TEMPLATE_NAME}
                else:
                    action = action_result["action"]
                    # Extract parameters correctly from the nested structure
                    params = action_result.get("parameters", {})
                    confidence = action_result.get("confidence", 0.0)
                    
                    yield f"Detected action: {self.actions[action]['description']} {action} (confidence: {confidence:.2f})\n"
                    yield f"Parameters: {json.dumps(params, indent=2)}\n\n"
                
                # HANDLE DIFFERENT ACTIONS
                
                # 1. Generate a full document
                if action == "write_full_doc":
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
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Generating document on '{topic}'", "done": False}
                        })
                    
                    yield f"Generating full document on '{topic}' using template: {template['name']}\n\n"
                    
                    # Initialize document storage
                    document = {
                        "topic": topic,
                        "template": template,
                        "model_id": llm_model_id,
                        "sections": {},
                        "created_at": time.time(),
                        "updated_at": time.time()
                    }
                    
                    # Generate each section
                    for section in template["sections"]:
                        section_name = section["title"]
                        section_key = section["name"]
                        
                        # Status update for section
                        if __event_emitter__:
                            await __event_emitter__({
                                "type": "status",
                                "data": {"description": f"Generating section: {section_name}", "done": False}
                            })
                        
                        yield f"Generating section: {section_name}...\n"
                        
                        # Initialize empty content for this section
                        section_content = ""
                        has_error = False
                        
                        # Stream the section generation
                        async for chunk in self.generate_section(client, llm_model_id, section, topic, headers):
                            if "error" in chunk:
                                yield f"Failed to generate section '{section_name}'. Moving to next section...\n"
                                has_error = True
                                break
                            elif "content_piece" in chunk:
                                # Accumulate content
                                section_content += chunk["content_piece"]
                                # Stream content piece to user within the thinking block
                                yield f"{chunk['content_piece']}"
                            elif "status" in chunk and chunk["status"] == "complete":
                                # Section completed
                                document["sections"][section_key] = chunk.get("content", section_content)
                                yield f"\nCompleted section: {section_name}\n"
                        
                        # If no error occurred and we didn't get a completion status
                        if not has_error and section_key not in document["sections"] and section_content:
                            document["sections"][section_key] = section_content
                            yield f"\nCompleted section: {section_name}\n"
                    
                    # Store the document
                    self.documents[chat_id] = document
                    
                    # End the thinking process
                    yield "\nCompiling final document...\n"
                    yield "</think>\n\n"
                    
                    # Now compile the final document (outside the thinking tags)
                    yield self.format_document(document)
                
                # 2. Edit a specific section
                elif action == "edit_section":
                    section_name = params.get("section_name", "")
                    instructions = params.get("instructions", "Improve this section")
                    
                    # Check if we have a document
                    if chat_id not in self.documents:
                        yield "No document found to edit. Please generate a document first."
                        yield "</think>\n\n"
                        return
                    
                    document = self.documents[chat_id]
                    llm_model_id = document["model_id"]
                    
                    # Find section by exact match first
                    section_found = False

                    # return all section list:
                    yield f"Section list: " 
                    for section in document["template"]["sections"]:
                        yield f"{section['name']} ({section['title']})\n"
                    
                    for section in document["template"]["sections"]:
                        if section["name"].lower() == section_name.lower():
                            section_name = section["name"]
                            section_found = True
                            break
                    
                    # If not found, try partial match
                    if not section_found:
                        # List all available sections for debugging
                        yield f"Section '{section_name}' not found by exact match. Trying partial match...\n"
                        yield "Available sections: " + ", ".join([f"{s['name']} ({s['title']})" for s in document["template"]["sections"]])
                        
                        # Try to find by title or name using partial match
                        best_match = None
                        best_score = 0
                        for section in document["template"]["sections"]:
                            # Check both name and title for matches
                            name_score = 0
                            if section_name.lower() in section["name"].lower():
                                name_score = len(section_name) / len(section["name"])
                            
                            title_score = 0
                            if section_name.lower() in section["title"].lower():
                                title_score = len(section_name) / len(section["title"])
                            
                            score = max(name_score, title_score)
                            if score > best_score:
                                best_score = score
                                best_match = section["name"]
                        
                        if best_match and best_score > 0.3:  # Threshold for acceptable match
                            section_name = best_match
                            section_found = True
                            yield f"Found closest matching section: {section_name}\n"
                    
                    if not section_found or section_name not in document["sections"]:
                        yield f"Section '{section_name}' not found in the document.\n"
                        yield "Available sections: " + ", ".join([s["title"] for s in document["template"]["sections"]])
                        yield "</think>\n\n"
                        
                        # Return a helpful message to the user
                        yield f"# Section Not Found\n\n"
                        yield f"I couldn't find a section named '{section_name}' in your document. Here are the available sections:\n\n"
                        for section in document["template"]["sections"]:
                            yield f"- **{section['title']}** (section name: {section['name']})\n"
                        yield f"\nPlease try again with one of these section names.\n"
                        return
                    
                    # Get section title
                    section_title = next((s["title"] for s in document["template"]["sections"] 
                                        if s["name"] == section_name), section_name)
                    
                    yield f"Editing section: {section_title} ({section_name}) with instructions: {instructions}\n\n"
                    
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Editing section: {section_title}", "done": False}
                        })
                    
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
                            document["sections"][section_name] = chunk.get("content", section_content)
                            document["updated_at"] = time.time()
                            yield f"\nCompleted editing section: {section_title}\n"
                    
                    # End thinking and show the entire document
                    yield "</think>\n\n"
                    
                    # Format and display the document, highlighting the edited section
                    yield self.format_document(document, highlight_section=section_name)
                
                # 3. Add a new section
                elif action == "add_section":
                    section_title = params.get("section_title", "New Section")
                    instructions = params.get("instructions", "Add content for this section")
                    
                    # Check if we have a document
                    if chat_id not in self.documents:
                        yield "No document found to modify. Please generate a document first."
                        yield "</think>\n\n"
                        return
                    
                    document = self.documents[chat_id]
                    llm_model_id = document["model_id"]
                    
                    yield f"Adding new section: {section_title} with instructions: {instructions}\n\n"
                    
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Adding section: {section_title}", "done": False}
                        })
                    
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
                                document["sections"][section_name] = section_content
                                document["template"]["sections"].append(section_metadata)
                                document["updated_at"] = time.time()
                                yield f"\nCompleted adding section: {section_title}\n"
                    
                    # End thinking and show the new section
                    yield "</think>\n\n"
                    
                    if section_name and section_name in document["sections"]:
                        yield f"# New Section Added: {section_title}\n\n"
                        yield f"{document['sections'][section_name]}\n\n"
                        yield f"\n---\nSection added at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['updated_at']))}\n"
                    else:
                        yield "Failed to add the new section."
                
                # 4. List all sections in the document
                elif action == "list_sections":
                    # Check if we have a document
                    if chat_id not in self.documents:
                        yield "No document found. Please generate a document first."
                        yield "</think>\n\n"
                        return
                    
                    document = self.documents[chat_id]
                    
                    yield f"Listing sections for document on topic: {document['topic']}\n\n"
                    
                    # End thinking
                    yield "</think>\n\n"
                    
                    yield f"# Document Sections\n\n"
                    yield f"**Topic:** {document['topic']}\n"
                    yield f"**Template:** {document['template']['name']}\n"
                    yield f"**Created:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['created_at']))}\n"
                    yield f"**Last Updated:** {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(document['updated_at']))}\n\n"
                    
                    yield "## Available Sections:\n\n"
                    
                    for section in document["template"]["sections"]:
                        section_name = section["name"]
                        section_title = section["title"]
                        if section_name in document["sections"]:
                            word_count = len(document["sections"][section_name].split())
                            yield f"- **{section_title}** ({word_count} words)\n"
                        else:
                            yield f"- {section_title} (not generated)\n"
                
                # Default case - unrecognized action
                else:
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
                        yield "- Add a section: *Add a section called 'Future Work' discussing potential next steps*\n"
                        yield "- List sections: *Show me the sections in my document*\n"
                    else:
                        yield "# Document Generator\n\n"
                        yield "No document has been created yet. You can:\n\n"
                        yield "- Generate a new document: *Write a document about [topic]*\n"
            
            # Send completion status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Operation completed", "done": True}
                })
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(f"Document operation error: {error_msg}")
            
            # Send error status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
                
            # If we're in the middle of thinking, close the tag
            yield "</think>\n\n"
            yield f"Error: {error_msg}"

    def format_document(self, document, highlight_section=None, include_metadata=True):
        """
        Format a document as markdown for display to the user.
        
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
        
        # Display each section with its content
        for section in document["template"]["sections"]:
            section_key = section["name"]
            if section_key in document["sections"]:
                # Highlight the edited section if specified
                if section_key == highlight_section:
                    output.append(f"## {section['title']} (Updated)\n")
                else:
                    output.append(f"## {section['title']}\n")
                
                output.append(f"{document['sections'][section_key]}\n\n")
        
        # Add metadata if requested
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