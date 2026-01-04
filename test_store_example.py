import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException
from d1.store_example import chat, ChatRequest, ChatResponse, get_runnable_config
from d1.store_example import do_message


class TestChatFunction:
    """Test suite for the chat function"""

    def test_chat_successful_response(self):
        """Test successful chat response with valid inputs"""
        # Arrange
        session_id = "test-session-123"
        request = ChatRequest(msg="Hello, how are you?")
        
        # Mock the do_message.invoke method to return a successful response
        mock_result = "I'm doing well, thank you!"
        with patch.object(do_message, 'invoke', return_value=mock_result):
            # Act
            response = chat(session_id, request)
            
            # Assert
            assert response.code == 200
            assert response.msg == mock_result
            do_message.invoke.assert_called_once_with(
                request.model_dump(),
                config=get_runnable_config(session_id)
            )

    def test_chat_empty_session_id(self):
        """Test chat with empty session_id should return error response"""
        # Arrange
        session_id = ""
        request = ChatRequest(msg="Hello")
        
        # Act
        response = chat(session_id, request)
        
        # Assert
        assert response.code == 500
        assert "Error:" in response.msg
        assert "assert" in response.msg or "Session ID is required" in response.msg

    def test_chat_none_session_id(self):
        """Test chat with None session_id should return error response"""
        # Arrange
        session_id = None
        request = ChatRequest(msg="Hello")
        
        # Act
        response = chat(session_id, request)
        
        # Assert
        assert response.code == 500
        assert "Error:" in response.msg

    def test_chat_do_message_exception(self):
        """Test chat when do_message.invoke raises an exception"""
        # Arrange
        session_id = "test-session-456"
        request = ChatRequest(msg="Test message")
        
        # Mock do_message.invoke to raise an exception
        error_message = "Model service unavailable"
        with patch.object(do_message, 'invoke', side_effect=Exception(error_message)):
            # Act
            response = chat(session_id, request)
            
            # Assert
            assert response.code == 500
            assert "Error:" in response.msg
            assert error_message in response.msg

    def test_chat_different_message_types(self):
        """Test chat with various message types and lengths"""
        test_cases = [
            ("Short message", "Hi"),
            ("Long message", "This is a very long message with many words to test the chat functionality with extended input that should work properly"),
            ("Question", "What is the capital of France?"),
            ("Empty message", ""),
            ("Special characters", "Hello! @#$%^&*()_+{}|:\"<>?[]\\;',./"),
        ]
        
        for desc, message in test_cases:
            with patch.object(do_message, 'invoke', return_value=f"Response to: {message}"):
                # Act
                response = chat("test-session", ChatRequest(msg=message))
                
                # Assert
                assert response.code == 200
                assert f"Response to: {message}" in response.msg

    def test_chat_session_id_boundary_values(self):
        """Test chat with various session_id formats and lengths"""
        test_session_ids = [
            ("uuid_format", "123e4567-e89b-12d3-a456-426614174000"),
            ("short_id", "abc"),
            ("long_id", "x" * 1000),
            ("numeric_id", "1234567890"),
            ("special_chars_id", "session!@#$%^&*()_+"),
        ]
        
        for desc, session_id in test_session_ids:
            with patch.object(do_message, 'invoke', return_value="Test response"):
                # Act
                response = chat(session_id, ChatRequest(msg="Test"))
                
                # Assert
                assert response.code == 200
                do_message.invoke.assert_called_with(
                    {"msg": "Test"},
                    config=get_runnable_config(session_id)
                )

    def test_chat_error_handling_different_exceptions(self):
        """Test chat handles different types of exceptions properly"""
        exception_types = [
            ValueError("Value error occurred"),
            RuntimeError("Runtime error occurred"),
            ConnectionError("Connection failed"),
            TypeError("Type mismatch"),
        ]
        
        for exception in exception_types:
            with patch.object(do_message, 'invoke', side_effect=exception):
                # Act
                response = chat("test-session", ChatRequest(msg="Test"))
                
                # Assert
                assert response.code == 500
                assert "Error:" in response.msg
                assert str(exception) in response.msg

    def test_chat_configuration_consistency(self):
        """Test that the same session_id always gets consistent configuration"""
        session_id = "consistent-session"
        request = ChatRequest(msg="Test consistency")
        
        with patch.object(do_message, 'invoke', return_value="Response") as mock_invoke:
            # Call multiple times with same session_id
            for _ in range(3):
                response = chat(session_id, request)
                assert response.code == 200
            
            # Verify all calls used the same config
            assert mock_invoke.call_count == 3
            for call in mock_invoke.call_args_list:
                args, kwargs = call
                expected_config = get_runnable_config(session_id)
                assert kwargs['config'] == expected_config

    def test_chat_response_model_validation(self):
        """Test that response always conforms to ChatResponse model"""
        session_id = "validation-test"
        request = ChatRequest(msg="Test validation")
        
        with patch.object(do_message, 'invoke', return_value="Valid response"):
            response = chat(session_id, request)
            
            # Verify response is a valid ChatResponse instance
            assert isinstance(response, ChatResponse)
            assert hasattr(response, 'code')
            assert hasattr(response, 'msg')
            assert isinstance(response.code, int)
            assert isinstance(response.msg, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])