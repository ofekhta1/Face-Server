from typing import Any, Callable, Union, List
from Shared.models.processing_message import ProcessingMessage
from prometheus_client import Gauge, Histogram

class BaseProcessingQueue:
    """
    Abstract base class defining the interface for a processing queue.
    
    This class provides a consistent interface for queues that can:
    - Enqueue messages
    - Consume messages 
    - Handle message processing
    - Track queue metrics
    """

    def enqueue(self, message: Union[ProcessingMessage, List[ProcessingMessage]]) -> bool:
        """
        Enqueue a single message or a list of messages.
        
        :param message: Single message or list of messages to enqueue
        :return: True if enqueue is successful, False otherwise
        """
        raise NotImplementedError(f"enqueue not implemented in {self.__class__.__name__}");

    def start_consuming(self, callback: Callable[[Any, int], None]):
        """
        Start consuming messages using the provided callback.
        
        :param callback: Function to process messages with signature (message, delivery_tag)
        """
        raise NotImplementedError(f"start_consuming not implemented in {self.__class__.__name__}");

    def complete_task(self, delivery_tag: int):
        """
        Mark a task as successfully completed.
        
        :param delivery_tag: Unique identifier for the message
        """
        raise NotImplementedError(f"complete_task not implemented in {self.__class__.__name__}");

    def fail_message(self, delivery_tag: int, message: ProcessingMessage = None, **kwargs) -> bool:
        """
        Handle a failed message, with optional requeue or move to dead-letter queue.
        
        :param delivery_tag: Unique identifier for the message
        :param message: Optional message object (useful for some queue implementations)
        :param kwargs: Additional implementation-specific arguments
        :return: True if message handling was successful
        """
        raise NotImplementedError(f"fail_message not implemented in {self.__class__.__name__}");

    def close(self):
        """
        Close the queue and release any resources.
        
        This method should:
        - Stop consuming messages
        - Close connections
        - Clean up resources
        """
        pass

    def update(self):
        """
        Update queue metrics.
        
        This method should:
        - Refresh queue length gauge
        - Perform any necessary periodic updates
        """
        pass

    def get_queue_length(self) -> int:
        """
        Get the current number of messages in the queue.
        
        :return: Number of messages in the queue
        """
        raise NotImplementedError(f"get_queue_length not implemented in {self.__class__.__name__}");

    def format_callback(self, callback: Callable[[Any, int], None], message: Any, delivery_tag: int):
        """
        Default implementation of callback formatting.
        
        :param callback: Function to process the message
        :param message: Message to be processed
        :param delivery_tag: Unique identifier for the message
        """
        # Common implementation that can be used by subclasses
        raise NotImplementedError(f"format_callback not implemented in {self.__class__.__name__}");