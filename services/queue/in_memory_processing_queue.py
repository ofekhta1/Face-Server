from queue import Queue
from prometheus_client import Gauge, Histogram
from time import time

class InMemoryProcessingQueue:
    # Define Prometheus metrics
    queue_length_gauge = Gauge('inmemory_face_queue_length', 'Current number of messages in the in-memory queue')
    message_processing_time_histogram = Histogram('inmemory_face_queue_processing_time_seconds', 'Time spent processing a message')

    def __init__(self):
        self.queue = Queue()
        self.delivery_tag_counter = 0  # Simulate delivery tags

    def enqueue(self, message):
        if isinstance(message, list):
            for msg in message:
                self.queue.put((msg, self.delivery_tag_counter))
                self.delivery_tag_counter += 1
        else:
            self.queue.put((message, self.delivery_tag_counter))
            self.delivery_tag_counter += 1

        # Update the queue length gauge metric
        self.queue_length_gauge.set(self.get_queue_length())

    def start_consuming(self, callback):
        """ Simulate consuming by using a callback for processing messages """
        while True:
            result = self.queue.get()
            message, delivery_tag=result
             # Start timer for message processing
            start_time = time()

            self.format_callback(callback, message, delivery_tag)

            # Record processing time
            processing_time = time() - start_time
            self.message_processing_time_histogram.observe(processing_time)

    def format_callback(self, callback, message, delivery_tag):
        """ Format and call the callback"""
        callback(message, delivery_tag)

    def complete_task(self, delivery_tag):

        """ Simulate acknowledging a task (no-op in-memory) """
        self.queue.task_done()
        self.queue_length_gauge.set(self.get_queue_length())
    def close(self):
        """ Simulate closing the queue (no-op in-memory) """
        pass
    def update(self):
        """
        Simulate updating the queue length (no-op in-memory).

        This method is used to update the queue length metric in Prometheus.
        Since the queue is in-memory, this operation is a no-op. The queue length
        is automatically updated whenever a message is enqueued or dequeued.

        Parameters:
        None

        Returns:
        None
        """
        pass; 
    def get_queue_length(self):
        """ Return the current queue length """
        return self.queue.qsize()