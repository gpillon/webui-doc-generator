"""
title: Document Generator
author: Unknown
author_url: Unknown
git_url: https://github.com/gpillon/document_generator_pipe
description: Document Generator
required_open_webui_version: 0.4.3
requirements: fastapi, pydantic, httpx
version: 0.4.3
licence: MIT
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union, Generator, Iterator
import json
import httpx
import asyncio
import logging
import time
from fastapi import Request

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
            description="API Host URL"
        )
        API_KEY: str = Field(
            default="",
            description="API Key for the model provider"
        )
        MODEL_ID: str = Field(
            default="gpt-3.5-turbo",
            description="Model ID to use for generation (comma-separated for multiple models)"
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

    def __init__(self):
        self.valves = self.Valves()
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

    async def pipe(self, body: dict, __event_emitter__=None) -> Union[str, Generator, Iterator]:
        """
        Generate a document by calling the external API for each section.
        
        Args:
            body: Contains user message, model ID, and messages history
            __event_emitter__: Optional function to emit status events
            
        Returns:
            A document in markdown format
        """
        # Extract data from the request body
        logger.debug(f"Document generation request received: {body}")
        
        # Extract required information from body
        user_message = ""
        model_id = ""
        messages = []
        
        # Extract user message (topic) from body
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
        
        # Log for debugging
        logger.debug(f"Extracted model: {model_id}, User message: {user_message}")
        
        # Validate API key
        if not self.valves.API_KEY:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Error: API key not configured", "done": True}
                })
            yield "Error: API key not configured. Please set the API_KEY valve."
            return
            
        # Get the topic from the user message
        topic = user_message.strip()
        if not topic:
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Error: No topic provided", "done": True}
                })
            yield "Error: No topic provided. Please specify a topic for the document."
            return
        
        # Parse the model and template from the selected option
        parts = model_id.split(":", 1)
        if len(parts) != 2:
            # Try using the first available template with the model
            llm_model_id = model_id
            template_id = self.valves.TEMPLATE_NAME
        else:
            llm_model_id, template_id = parts
        
        if template_id not in self.templates:
            template_id = self.valves.TEMPLATE_NAME
            
        template = self.templates[template_id]
        
        # Prepare request headers
        headers = {
            "Authorization": f"Bearer {self.valves.API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openwebui.com/",
            "X-Title": "Open WebUI",
        }
        
        # Log the actual model ID being used
        logger.debug(f"Using model ID: {llm_model_id}")
        
        try:
            # Stream the document generation process
            document_parts = {}
            errors = []
            
            # Start the thinking process
            yield "<think>\n"
            
            # First, return a status message
            yield f"Generating document on '{topic}' using template: {template['name']}\n\n"
            
            # Send status update
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": f"Generating document on '{topic}'", "done": False}
                })
            
            # Create sections one by one with improved timeout and connection settings
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            timeout = httpx.Timeout(self.valves.TIMEOUT, connect=5.0)
            
            async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
                for section in template["sections"]:
                    section_name = section["title"]
                    section_key = section["name"]
                    
                    # Status update for section
                    if __event_emitter__:
                        await __event_emitter__({
                            "type": "status",
                            "data": {"description": f"Generating section: {section_name}", "done": False}
                        })
                    
                    yield f" ✍️ Generating section: {section_name}...\n"
                    
                    # Initialize empty content for this section
                    section_content = ""
                    has_error = False
                    
                    # Stream the section generation
                    async for chunk in self.generate_section(client, llm_model_id, section, topic, headers):
                        if "error" in chunk:
                            errors.append(f"Error in section '{section_key}': {chunk['error']}")
                            has_error = True
                            yield f"Failed to generate section '{section_name}'. Moving to next section...\n"
                            break
                        elif "content_piece" in chunk:
                            # Accumulate content
                            section_content += chunk["content_piece"]
                            # Stream content piece to user within the thinking block
                            yield f"{chunk['content_piece']}"
                        elif "status" in chunk and chunk["status"] == "complete":
                            # Section completed
                            document_parts[section_key] = chunk.get("content", section_content)
                            yield f"\nCompleted section: {section_name}\n"
                    
                    # If no error occurred and we didn't get a completion status
                    if not has_error and section_key not in document_parts and section_content:
                        document_parts[section_key] = section_content
                        yield f"\nCompleted section: {section_name}\n"
            
            # If there were errors, report them within the thinking block
            if errors:
                yield "\nGeneration Errors:\n"
                for error in errors:
                    yield f"* {error}\n"
                
            # Send near-completion status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Compiling final document...", "done": False}
                })
                
            yield "\nCompiling final document...\n"
            
            # End the thinking process
            yield "</think>\n\n"
            
            # Now compile the final document (outside the thinking tags)
            yield "# Generated Document\n\n"
            
            for section in template["sections"]:
                section_key = section["name"]
                if section_key in document_parts:
                    section_content = document_parts[section_key]
                    
                    if "title" in section:
                        yield f"## {section['title']}\n\n"
                    yield f"{section_content}\n\n"
                    
            # Return metadata
            yield f"\n---\nDocument generated using model: {llm_model_id}, template: {template['name']}\n"
            
            # Send completion status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": "Document generation completed", "done": True}
                })
                
        except Exception as e:
            error_msg = f"Error generating document: {str(e)}"
            logger.error(f"Document generation error: {error_msg}")
            
            # Send error status
            if __event_emitter__:
                await __event_emitter__({
                    "type": "status",
                    "data": {"description": error_msg, "done": True}
                })
                
            # If we're in the middle of thinking, close the tag
            yield "</think>\n\n"
            yield f"Error: {error_msg}"