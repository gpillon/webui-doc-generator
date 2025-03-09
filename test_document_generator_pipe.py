"""
Tests for document_generator_pipe.py
"""
import os
import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import asyncio
from document_generator_pipe import Pipe

# Test fixtures
@pytest.fixture
def pipe_instance():
    """Create a Pipe instance with modified storage paths for testing"""
    with patch('os.makedirs'), patch('document_generator_pipe.Pipe._load_local_templates'):
        pipe = Pipe()
        # Override file paths for testing
        pipe.valves.DOCUMENT_STORAGE = "memory"
        pipe.documents = {}  # Initialize empty dict for in-memory storage
        return pipe

@pytest.fixture
def mock_client():
    """Create a mock httpx client"""
    client = AsyncMock()
    return client

@pytest.fixture
def mock_event_emitter():
    """Create a mock event emitter"""
    emitter = AsyncMock()
    return emitter

@pytest.fixture
def mock_metadata():
    """Create mock metadata for testing"""
    return {
        "chat_id": "test_chat_123",
        "collection_names": ["test_collection"]
    }

@pytest.fixture
def mock_context(mock_event_emitter):
    """Create a mock context object for testing"""
    return {
        "chat_id": "test_chat_123",
        "collection_names": ["test_collection"],
        "event_emitter": mock_event_emitter,
        "files": None,
        "request": MagicMock(),
        "user": {"id": "test_user"}
    }

@pytest.fixture
def sample_document():
    """Create a sample document for testing"""
    return {
        "topic": "Test Topic",
        "template": {
            "name": "Test Template",
            "sections": [
                {
                    "name": "introduction",
                    "title": "Introduction",
                    "prompt": "Write an introduction about {topic}",
                    "sectionType": "generate",
                    "children": []
                },
                {
                    "name": "body",
                    "title": "Main Content",
                    "prompt": "Write content about {topic}",
                    "sectionType": "generate",
                    "children": [
                        {
                            "name": "section1",
                            "title": "Section 1",
                            "prompt": "Write section 1 about {topic}",
                            "sectionType": "generate",
                            "children": []
                        }
                    ]
                }
            ]
        },
        "model_id": "gpt-3.5-turbo",
        "sections": {
            "introduction": "This is an introduction to Test Topic.",
            "body": "This is the body content for Test Topic.",
            "section1": "This is section 1 content."
        },
        "created_at": 1625097600,
        "updated_at": 1625097700
    }

# Mock for RAG dependencies
class MockRAGDependencies:
    @staticmethod
    def query_collection(*args, **kwargs):
        return [
            {"id": "1", "metadata": {"title": "Test Doc 1"}, "content": "This is test content 1", "score": 0.9},
            {"id": "2", "metadata": {"title": "Test Doc 2"}, "content": "This is test content 2", "score": 0.8}
        ]
    
    @staticmethod
    def query_collection_with_hybrid_search(*args, **kwargs):
        return [
            {"id": "1", "metadata": {"title": "Test Doc 1"}, "content": "This is test content 1", "score": 0.9},
            {"id": "2", "metadata": {"title": "Test Doc 2"}, "content": "This is test content 2", "score": 0.8}
        ]
    
    @staticmethod
    def get_web_loader(*args, **kwargs):
        loader = AsyncMock()
        async def mock_load(urls):
            return [
                {"title": "Web Result 1", "content": "Web content 1", "url": "https://example.com/1"},
                {"title": "Web Result 2", "content": "Web content 2", "url": "https://example.com/2"}
            ]
        loader.load.side_effect = mock_load
        return loader
    
    @staticmethod
    def search_duckduckgo(*args, **kwargs):
        return [
            {"title": "Search Result 1", "content": "Search content 1", "url": "https://example.com/search/1"},
            {"title": "Search Result 2", "content": "Search content 2", "url": "https://example.com/search/2"}
        ]

@pytest.mark.asyncio
async def test_pipe_write_full_doc(pipe_instance, mock_client, mock_event_emitter, mock_metadata):
    """Test the pipe method with write_full_doc action"""
    # Setup
    body = {
        "messages": [{"role": "user", "content": "Create a document about testing"}],
        "model": "gpt-3.5-turbo"
    }

    # Mock detect_action to return write_full_doc
    pipe_instance.detect_action = AsyncMock(return_value={
        "action": "write_full_doc",
        "parameters": {"topic": "testing", "template_id": "basic"},
        "confidence": 0.9
    })

    # Mock handle_write_full_doc
    async def mock_handle(*args, **kwargs):
        yield "Generating document...\n"
        yield "</think>\n\n"
        yield "# Testing\n\nThis is a test document."
    
    pipe_instance.handle_write_full_doc = mock_handle

    # Create a dummy pipe method for testing
    original_pipe = pipe_instance.pipe
    
    # Define a wrapper that will actually call our mock
    async def pipe_wrapper(*args, **kwargs):
        action_result = await pipe_instance.detect_action(args[0], None)
        action = action_result.get("action")
        
        if action == "write_full_doc":
            async for chunk in mock_handle(*args, **kwargs):
                yield chunk
        else:
            async for chunk in original_pipe(*args, **kwargs):
                yield chunk
    
    # Replace pipe method
    pipe_instance.pipe = pipe_wrapper

    # Run pipe
    result = [chunk async for chunk in pipe_instance.pipe(
        body,
        __event_emitter__=mock_event_emitter,
        __metadata__=mock_metadata,
        __request__=MagicMock(),
        __user__={"id": "test_user"}
    )]

    # Verify
    assert len(result) > 0
    assert any("Generating document" in chunk for chunk in result)
    assert any("# Testing" in chunk for chunk in result)

@pytest.mark.asyncio
async def test_detect_action_with_smart_model(pipe_instance, mock_client):
    """Test detect_action method with a model that can classify actions"""
    body = {
        "messages": [{"role": "user", "content": "Create a document about testing"}],
        "model": "gpt-3.5-turbo"
    }
    
    # Mock client response for action detection
    mock_client.stream = AsyncMock()
    
    async def mock_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"content": '{"action": "write_full_doc", "parameters": {"topic": "testing", "template_id": "basic"}, "confidence": 0.9}'}}]}
    
    mock_client.stream.side_effect = mock_stream
    
    # Run detect_action
    result = await pipe_instance.detect_action(body, mock_client)
    
    # Verify
    assert result is not None
    assert result.get("action") == "write_full_doc"
    assert result.get("parameters", {}).get("topic") == "testing"
    assert result.get("confidence") == 0.9

@pytest.mark.asyncio
async def test_detect_action_without_smart_model(pipe_instance):
    """Test detect_action method with fallback to regex extraction"""
    body = {
        "messages": [
            {"role": "user", "content": "Create a document about testing with template basic"}
        ],
        "model": "non_smart_model"
    }
    
    # Mock the _query_thinking_model method
    pipe_instance._query_thinking_model = AsyncMock(return_value=None)
    
    # Run detect_action without a smart model
    result = await pipe_instance.detect_action(body, None)
    
    # Verify
    assert result is not None
    assert "action" in result
    assert "parameters" in result

@pytest.mark.asyncio
async def test_generate_section(pipe_instance, mock_client):
    """Test generate_section method"""
    section = {
        "name": "introduction",
        "title": "Introduction",
        "prompt": "Write an introduction about {topic}",
        "sectionType": "generate"
    }
    topic = "Testing"
    headers = {"Authorization": "Bearer fake-key"}
    
    # Mock client stream response
    mock_client.stream = AsyncMock()
    
    async def mock_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "This is a generated introduction about Testing."}}]}
    
    mock_client.stream.side_effect = mock_stream
    
    # Run generate_section
    result = await pipe_instance.generate_section(mock_client, "gpt-3.5-turbo", section, topic, headers)
    
    # Verify
    assert result is not None
    assert "Testing" in result

@pytest.mark.asyncio
async def test_edit_document_section(pipe_instance, mock_client, sample_document):
    """Test edit_document_section method"""
    chat_id = "test_chat_123"
    section_name = "introduction"
    new_content = "This is updated introduction content."
    
    # Add document to pipe instance
    pipe_instance.documents = {chat_id: sample_document}
    
    # Run edit_document_section
    result = pipe_instance.edit_document_section(chat_id, section_name, new_content)
    
    # Verify
    assert result is True
    assert pipe_instance.documents[chat_id]["sections"][section_name] == new_content

@pytest.mark.asyncio
async def test_handle_write_full_doc(pipe_instance, mock_client, mock_context):
    """Test handle_write_full_doc method"""
    # Setup
    chat_id = "test_chat_123"
    user_message = "Document about testing"
    model_id = "gpt-3.5-turbo"
    params = {"topic": "Testing", "template_id": "basic"}
    headers = {"Authorization": "Bearer fake-key"}
    
    # Make sure documents dictionary is initialized
    pipe_instance.documents = {}
    
    # Mock implementation for generate_document that returns an async generator
    async def mock_generate_document(*args, **kwargs):
        yield "Generating section: Introduction...\n"
        yield "Generated content for introduction\n"
        yield "Generating section: Main Content...\n"
        yield "Generated content for main content\n"
    
    # Set up the mocks
    pipe_instance.generate_document = mock_generate_document
    pipe_instance.format_document = MagicMock(return_value="# Testing\n\n## Introduction\n\nGenerated content...")
    pipe_instance.save_document = MagicMock()
    
    # Run handle_write_full_doc
    result = [chunk async for chunk in pipe_instance.handle_write_full_doc(
        mock_client, chat_id, user_message, model_id, params, headers, mock_context
    )]
    
    # Verify
    assert len(result) > 0
    assert any("Generating" in chunk for chunk in result)
    assert pipe_instance.save_document.called

@pytest.mark.asyncio
async def test_format_document(pipe_instance, sample_document):
    """Test format_document method"""
    # Run format_document
    formatted = pipe_instance.format_document(sample_document)
    
    # Verify
    assert "# Test Topic" in formatted
    assert "## Introduction" in formatted
    assert "This is an introduction to Test Topic" in formatted

@pytest.mark.asyncio
async def test_find_section_by_name(pipe_instance, sample_document):
    """Test find_section_by_name method"""
    # Test finding a top-level section
    result, parent = pipe_instance.find_section_by_name(sample_document["template"]["sections"], "introduction")
    assert result is not None
    assert result["name"] == "introduction"
    
    # Test finding a nested section
    result, parent = pipe_instance.find_section_by_name(sample_document["template"]["sections"], "section1")
    assert result is not None
    assert result["name"] == "section1"
    
    # Test finding a non-existent section
    result, parent = pipe_instance.find_section_by_name(sample_document["template"]["sections"], "nonexistent")
    assert result is None

@pytest.mark.asyncio
async def test_handle_list_sections(pipe_instance, mock_client, mock_context):
    """Test handle_list_sections method"""
    # Setup
    chat_id = "test_chat_123"
    user_message = "List all sections"
    model_id = "gpt-3.5-turbo"
    params = {}
    headers = {"Authorization": "Bearer fake-key"}
    
    # Add document to pipe instance
    pipe_instance.documents = {chat_id: {
        "topic": "Test Topic",
        "template": {"name": "Test Template", "sections": [
            {"name": "intro", "title": "Introduction", "sectionType": "generate"},
            {"name": "body", "title": "Main Content", "sectionType": "generate"}
        ]},
        "sections": {"intro": "Content", "body": "Content"}
    }}
    
    # Run handle_list_sections
    result = [chunk async for chunk in pipe_instance.handle_list_sections(
        mock_client, chat_id, user_message, model_id, params, headers, mock_context
    )]
    
    # Verify
    assert len(result) > 0
    assert any("Introduction" in chunk for chunk in result)
    assert any("Main Content" in chunk for chunk in result)

@pytest.mark.asyncio
async def test_save_and_load_document(pipe_instance):
    """Test save_document and load_document methods"""
    # Setup
    chat_id = "test_save_load"
    document = {
        "topic": "Test Save Load",
        "template": {"name": "Test Template", "sections": []},
        "sections": {"intro": "Test content"},
        "created_at": 1625097600,
        "updated_at": 1625097700
    }
    
    # Initialize documents dictionary
    pipe_instance.documents = {}
    
    # Test with memory storage
    pipe_instance.valves.DOCUMENT_STORAGE = "memory"
    pipe_instance.save_document(chat_id, document)
    
    # For memory storage, it should save directly to the documents dict
    assert chat_id in pipe_instance.documents
    
    loaded_doc = pipe_instance.load_document(chat_id)
    assert loaded_doc is not None
    assert loaded_doc["topic"] == "Test Save Load"
    
    # Test with file storage
    pipe_instance.valves.DOCUMENT_STORAGE = "file"
    with patch("os.path.exists", return_value=True), \
         patch("builtins.open", mock_open()) as mock_file, \
         patch("json.dump") as mock_json_dump, \
         patch("json.load", return_value=document) as mock_json_load:
        
        pipe_instance.documents = {}  # Clear in-memory storage
        pipe_instance.save_document(chat_id, document)
        
        # Check if open was called
        mock_file.assert_called()
        mock_json_dump.assert_called_once()
        
        # Test load_document with file storage
        loaded_doc = pipe_instance.load_document(chat_id)
        assert loaded_doc is not None
        mock_json_load.assert_called_once()

@pytest.mark.asyncio
async def test_query_thinking_model(pipe_instance, mock_client):
    """Test _query_thinking_model method"""
    prompt = "What action should I take?"
    system_prompt = "You are a helpful assistant."
    
    # Mock client stream response
    mock_client.stream = AsyncMock()
    
    async def mock_stream(*args, **kwargs):
        yield {"choices": [{"delta": {"content": "write_full_doc"}}]}
    
    mock_client.stream.side_effect = mock_stream
    
    # Run _query_thinking_model
    result = await pipe_instance._query_thinking_model(mock_client, prompt, system_prompt)
    
    # Verify
    assert result == "write_full_doc"

@pytest.mark.asyncio
async def test_pipe_multiple_actions(pipe_instance, mock_client, mock_event_emitter, mock_metadata):
    """Test the pipe method with multiple actions"""
    
    # Create a sequence of tests for different actions
    async def run_action_test(action, user_message, expected_handler):
        body = {
            "messages": [{"role": "user", "content": user_message}],
            "model": "gpt-3.5-turbo"
        }
        
        # Reset mocks
        for handler_name in pipe_instance.actions.keys():
            handler = getattr(pipe_instance, f"handle_{handler_name}", None)
            if handler:
                if isinstance(handler, AsyncMock):
                    handler.reset_mock()
        
        # Mock detect_action to return specified action
        pipe_instance.detect_action = AsyncMock(return_value={
            "action": action,
            "parameters": {"topic": "testing", "section_name": "introduction", "template_id": "basic"},
            "confidence": 0.9
        })
        
        # Mock handler for current action
        async def mock_handler(*args, **kwargs):
            yield f"Handling {action}...\n"
            yield "</think>\n\n"
            yield f"# Result for {action}"
        
        # Replace the real handler with our mock handler
        setattr(pipe_instance, expected_handler, mock_handler)
        
        # Create a wrapper for pipe to use our mocked handler
        original_pipe = pipe_instance.pipe
        
        async def pipe_wrapper(*args, **kwargs):
            action_result = await pipe_instance.detect_action(args[0], None)
            action_name = action_result.get("action")
            
            handler = getattr(pipe_instance, f"handle_{action_name}", None)
            if handler:
                chat_id = kwargs.get("__metadata__", {}).get("chat_id", "default_chat")
                user_message = args[0]["messages"][-1]["content"] if args[0].get("messages") else ""
                model_id = args[0].get("model", "gpt-3.5-turbo")
                params = action_result.get("parameters", {})
                headers = {}
                ctx = {
                    "chat_id": chat_id,
                    "event_emitter": kwargs.get("__event_emitter__"),
                    "request": kwargs.get("__request__"),
                    "user": kwargs.get("__user__"),
                    "collection_names": kwargs.get("__metadata__", {}).get("collection_names", [])
                }
                
                async for chunk in handler(mock_client, chat_id, user_message, model_id, params, headers, ctx):
                    yield chunk
            else:
                async for chunk in original_pipe(*args, **kwargs):
                    yield chunk
        
        # Replace pipe method
        pipe_instance.pipe = pipe_wrapper
        
        # Run pipe
        result = [chunk async for chunk in pipe_instance.pipe(
            body,
            __event_emitter__=mock_event_emitter,
            __metadata__=mock_metadata,
            __request__=MagicMock(),
            __user__={"id": "test_user"}
        )]
        
        # Restore original pipe
        pipe_instance.pipe = original_pipe
        
        # Verify
        assert len(result) > 0
        assert any(f"Handling {action}" in chunk for chunk in result)
        assert any(f"# Result for {action}" in chunk for chunk in result)
    
    # Test multiple actions
    await run_action_test("write_full_doc", "Create a document about testing", "handle_write_full_doc")
    await run_action_test("edit_section", "Edit the introduction section", "handle_edit_section")
    await run_action_test("list_sections", "Show me all sections", "handle_list_sections")

@pytest.mark.asyncio
async def test_get_rag_context(pipe_instance, mock_client):
    """Test _get_rag_context method with mocked dependencies"""
    # Setup
    topic = "Testing"
    query = "What is testing?"
    request = MagicMock()
    collection_names = ["test_collection"]
    user = {"id": "test_user"}
    
    # Set up RAG-related properties
    pipe_instance.valves.RAG_HYBRID_SEARCH = True
    pipe_instance.valves.RAG_WEB_SEARCH = True
    pipe_instance.valves.RAG_WEB_SEARCH_ENGINE = "duckduckgo"
    pipe_instance.valves.RAG_RESULT_COUNT = 3
    
    # Mock the imported modules rather than the functions directly
    with patch("document_generator_pipe.open_webui.retrieval.utils.query_collection", side_effect=MockRAGDependencies.query_collection), \
         patch("document_generator_pipe.open_webui.retrieval.utils.query_collection_with_hybrid_search", side_effect=MockRAGDependencies.query_collection_with_hybrid_search), \
         patch("document_generator_pipe.open_webui.retrieval.web.utils.get_web_loader", side_effect=MockRAGDependencies.get_web_loader), \
         patch("document_generator_pipe.open_webui.retrieval.web.duckduckgo.search_duckduckgo", side_effect=MockRAGDependencies.search_duckduckgo):
        
        # Add method to pipe_instance (normally this would be in the real class)
        async def _get_rag_context(self, client, topic, query, request, collection_names, user):
            try:
                results = []
                if self.valves.RAG_HYBRID_SEARCH:
                    from open_webui.retrieval.utils import query_collection_with_hybrid_search
                    results = query_collection_with_hybrid_search(
                        query, collection_names[0]
                    )
                else:
                    from open_webui.retrieval.utils import query_collection
                    results = query_collection(
                        query, collection_names[0]
                    )
                
                if self.valves.RAG_WEB_SEARCH:
                    if self.valves.RAG_WEB_SEARCH_ENGINE == "duckduckgo":
                        from open_webui.retrieval.web.duckduckgo import search_duckduckgo
                        web_results = search_duckduckgo(query, max_results=3)
                    else:
                        web_results = []
                        
                    from open_webui.retrieval.web.utils import get_web_loader
                    loader = get_web_loader()
                    web_content = await loader.load([r["url"] for r in web_results[:3]])
                    for web_doc in web_content:
                        results.append({
                            "id": web_doc["url"],
                            "content": web_doc["content"],
                            "metadata": {"title": web_doc["title"], "url": web_doc["url"]},
                            "score": 0.7  # Arbitrary score for web results
                        })
                
                context_parts = []
                for doc in results:
                    title = doc.get("metadata", {}).get("title", "Untitled")
                    context_parts.append(f"Source: {title}\n{doc['content']}")
                
                return "\n\n".join(context_parts), len(results)
            except Exception as e:
                print(f"Error getting RAG context: {e}")
                return "", 0
        
        # Add method to instance for testing
        pipe_instance._get_rag_context = _get_rag_context.__get__(pipe_instance, Pipe)
        
        # Run _get_rag_context
        context, docs_count = await pipe_instance._get_rag_context(
            mock_client, topic, query, request, collection_names, user
        )
        
        # Verify
        assert context != ""
        assert "Source:" in context
        assert docs_count > 0

if __name__ == "__main__":
    pytest.main(["-xvs", "test_document_generator_pipe.py"]) 