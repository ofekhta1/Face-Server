from queue import Queue
from prometheus_client import Gauge, Histogram
from time import time
from typing import Any, Callable, Union, List
from Shared.services.queue.base_processing_queue import BaseProcessingQueue
from Shared.models.processing_message import ProcessingMessage

class InMemoryProcessingQueue(BaseProcessingQueue):
    # Define Prometheus metrics
    queue_length_gauge = Gauge('inmemory_face_queue_length', 'Current number of messages in the in-memory queue')
    message_processing_time_histogram = Histogram('inmemory_face_queue_processing_time_seconds', 'Time spent processing a message')

    def __init__(self):
        """
        Initialize the in-memory processing queue.
        Creates a Queue and initializes a delivery tag counter.
        """
        self.queue = Queue()
        self.delivery_tag_counter = 0  # Simulate delivery tags
        self.message_store = {}  # Store messages for potential fail operation

    def enqueue(self, message: Union[ProcessingMessage, List[ProcessingMessage]]) -> bool:
        """
        Enqueue a single message or a list of messages.
        
        :param message: Single message or list of messages to enqueue
        :return: True if enqueue is successful
        """
        try:
            if isinstance(message, list):
                for msg in message:
                    # Store message with delivery tag
                    self.message_store[self.delivery_tag_counter] = msg
                    self.queue.put((msg, self.delivery_tag_counter))
                    self.delivery_tag_counter += 1
            else:
                # Store message with delivery tag
                self.message_store[self.delivery_tag_counter] = message
                self.queue.put((message, self.delivery_tag_counter))
                self.delivery_tag_counter += 1
            
            # Update the queue length gauge metric
            self.queue_length_gauge.set(self.get_queue_length())
            return True
        except Exception as e:
            print(f"Error enqueueing message: {e}")
            return False

    def start_consuming(self, callback: Callable[[Any, int], None]):
        """
        Simulate consuming messages by using a callback for processing.
        
        :param callback: Function to process messages
        """
        while True:
            try:
                # Get message and delivery tag
                result = self.queue.get()
                message, delivery_tag = result

                # Start timer for message processing
                start_time = time()
                
                # Process message
                self.format_callback(callback, message, delivery_tag)
                
                # Record processing time
                processing_time = time() - start_time
                self.message_processing_time_histogram.observe(processing_time)
            
            except Exception as e:
                print(f"Error in start_consuming: {e}")

    def format_callback(self, callback: Callable[[Any, int], None], message: Any, delivery_tag: int):
        """
        Format and call the callback function.
        
        :param callback: Function to process the message
        :param message: Message to be processed
        :param delivery_tag: Unique identifier for the message
        """
        callback(message, delivery_tag)

    def complete_task(self, delivery_tag: int):
        """
        Simulate acknowledging a task.
        
        :param delivery_tag: Unique identifier for the message
        """
        self.queue.task_done()
        
        # Remove message from store
        if delivery_tag in self.message_store:
            del self.message_store[delivery_tag]
        
        # Update queue length gauge
        self.queue_length_gauge.set(self.get_queue_length())

    def fail_message(self, delivery_tag: int, max_retries: int = 3):
        """
        Handle a failed message.
        
        :param delivery_tag: Unique identifier for the message
        :param max_retries: Maximum number of retry attempts
        :return: True if message handling was successful
        """
        if delivery_tag not in self.message_store:
            print(f"No message found with delivery tag: {delivery_tag}")
            return False

        try:
            # Retrieve the message
            message = self.message_store[delivery_tag]
            
            # Remove from current position
            self.queue.task_done()
            del self.message_store[delivery_tag]

            # Re-enqueue with a new delivery tag
            if hasattr(message, 'retry_count'):
                message.retry_count = getattr(message, 'retry_count', 0) + 1
            
            # Check if max retries exceeded
            if getattr(message, 'retry_count', 0) < max_retries:
                self.enqueue(message)
                print(f"Message requeued. Retry count: {getattr(message, 'retry_count', 1)}")
                return True
            else:
                print(f"Message exceeded max retries. Dropping message.")
                return False
        
        except Exception as e:
            print(f"Error failing message: {e}")
            return False

    def close(self):
        """
        Simulate closing the queue (no-op in-memory).
        """
        pass

    def update(self):
        """
        Update the queue length metric.
        In an in-memory queue, this is typically a no-op as 
        the length is automatically tracked.
        """
        self.queue_length_gauge.set(self.get_queue_length())

    def get_queue_length(self) -> int:
        """
        Return the current queue length.
        
        :return: Number of items in the queue
        """
        return self.queue.qsize()