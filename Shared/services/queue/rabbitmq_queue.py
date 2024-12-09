from Shared.models.processing_message import ProcessingMessage
import pika
import threading
import json
from time import time,sleep
from functools import partial
from prometheus_client import Gauge, Histogram


class RabbitMQQueue:
    queue_length_gauge = Gauge('face_queue_length', 'Current number of messages in the queue')
    message_processing_time_histogram = Histogram('message_processing_time_seconds', 'Time spent processing a message')

    def __init__(self, host, port, queue_name="faces_queue", max_retries=3, retry_delay=1):
        self.queue_name = queue_name
        self.fail_queue = queue_name+"_failed"
        self.host = host
        self.port = port
        self.consumer_thread = None
        self.should_stop = threading.Event()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.connection_params = pika.ConnectionParameters(host=self.host, port=self.port)
        
        # Connection and channel for consuming
        self.consume_connection = self._create_connection()
        self.consume_channel = self._create_channel(self.consume_connection)
        
        # Connection and channel for publishing
        self.publish_connection = self._create_connection()
        self.publish_channel = self._create_channel(self.publish_connection)
        
        # Connection and channel for queue length checks
        self.length_connection = self._create_connection()
        self.length_channel = self._create_channel(self.length_connection)
        
        # Locks for thread-safe operations
        self.publish_lock = threading.Lock()
        self.length_lock = threading.Lock()

    def _create_connection(self):
        return pika.BlockingConnection(self.connection_params)

    def _create_channel(self, connection):
        channel = connection.channel()
        channel.queue_declare(queue=self.queue_name)
        return channel

  
    def enqueue(self, message: ProcessingMessage | list[ProcessingMessage]):
        retry_count = 0
        while retry_count < self.max_retries:
            with self.publish_lock:
                try:
                    if not self.publish_connection.is_open:
                        self.publish_connection = self._create_connection()
                        self.publish_channel = self._create_channel(self.publish_connection)
                    
                    if isinstance(message, list):
                        for msg in message:
                            self.publish_channel.basic_publish(exchange='', routing_key=self.queue_name, body=msg.model_dump_json())
                    else:
                        self.publish_channel.basic_publish(exchange='', routing_key=self.queue_name, body=message.model_dump_json())
                    
                    # If we reach here, the publish was successful
                    return True
                except (pika.exceptions.AMQPError, pika.exceptions.StreamLostError) as e:
                    print(f"Error during enqueue (attempt {retry_count + 1}/{self.max_retries}): {e}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        # Release the lock before sleeping
                        self.publish_lock.release()
                        sleep(self.retry_delay)
                        # Reacquire the lock before the next iteration
                        self.publish_lock.acquire()
                    else:
                        print("Max retries reached. Failed to enqueue message.")
                finally:
                    # Update the queue length gauge
                    self.queue_length_gauge.set(self.get_queue_length())

        if retry_count == self.max_retries:
            # Handle the case where all retries failed
            # You might want to log this event or take some other action
            print("Failed to enqueue message after all retries.")
            return False

    

    def deserialize_task(self, body):
        task_data = json.loads(body)
        return ProcessingMessage(**task_data)

    def format_callback(self, callback, ch, method, properties, body):
        message = self.deserialize_task(body)
        delivery_tag = method.delivery_tag
        start_time = time()
        callback(message, delivery_tag)
        processing_time = time() - start_time
        self.message_processing_time_histogram.observe(processing_time)

    def consume(self, callback):
        def on_message(ch, method, properties, body):
            # Wrap the callback and acknowledgment in a thread-safe way
            def process_message():
                try:
                    # Deserialize the message and invoke the user-provided callback
                    message = self.deserialize_task(body)
                    start_time = time()
                    callback(message, method.delivery_tag)  # Pass the delivery tag for acknowledgment
                    processing_time = time() - start_time
                    self.message_processing_time_histogram.observe(processing_time)

                except Exception as e:
                    print(f"Error processing message: {e}")
                    # Optionally, you could reject the message here instead of acking
                    ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

            # Add the message processing to the Pika thread-safe callback system
            self.consume_connection.add_callback_threadsafe(process_message)

        # Declare the consumer on the queue
        self.consume_channel.basic_qos(prefetch_count=2)  # Prevent over-prefetching
        self.consume_channel.basic_consume(queue=self.queue_name, on_message_callback=on_message, auto_ack=False)

        # Start consuming messages
        while not self.should_stop.is_set():
            try:
                self.consume_connection.process_data_events(time_limit=1)
            except pika.exceptions.AMQPError as e:
                print(f"AMQP error during consume: {e}")
                # Recreate connection and channel
                self.consume_connection = self._create_connection()
                self.consume_channel = self._create_channel(self.consume_connection)
                self.consume_channel.basic_qos(prefetch_count=2)
                self.consume_channel.basic_consume(queue=self.queue_name, on_message_callback=on_message, auto_ack=False)

    def start_consuming(self, callback):
        self.consumer_thread = threading.Thread(target=self.consume, args=(callback,))
        self.consumer_thread.start()

    def complete_task(self, delivery_tag):
        def ack_callback():
            self.consume_channel.basic_ack(delivery_tag=delivery_tag)
        self.consume_connection.add_callback_threadsafe(ack_callback)

    def fail_message(self, delivery_tag,message:ProcessingMessage, requeue=True):
        """
        Mark a message as failed.

        :param delivery_tag: Unique identifier of the message to fail
        :param requeue: Whether to requeue the message into the failed_queue (default: True)
        :return: True if message was successfully processed as failed, False otherwise
        """
        def fail_callback():
            if  requeue:

                self.consume_channel.basic_nack(
                    delivery_tag=delivery_tag, 
                    requeue=True, 
                )
                print(f"Message {delivery_tag} requeued. ")
            else:
                self.consume_channel.basic_publish(
                    exchange='',
                    routing_key=self.fail_queue,
                    body=message.model_dump_json(),
                    properties=pika.BasicProperties(
                        headers={
                        }
                    )
                )
                print(f"Message {delivery_tag} moved to failed queue")
                
                # Acknowledge the original message to remove it from the queue
                self.consume_channel.basic_ack(delivery_tag=delivery_tag)


            return True

        # Execute the fail callback in a thread-safe manner
        self.consume_connection.add_callback_threadsafe(fail_callback)
        return True


    def close(self):
        self.should_stop.set()
        if self.consumer_thread:
            self.consumer_thread.join()
        for conn in [self.consume_connection, self.publish_connection, self.length_connection]:
            if conn and conn.is_open:
                conn.close()

    def update(self):
        self.queue_length_gauge.set(self.get_queue_length())

    def get_queue_length(self):
        with self.length_lock:
            try:
                if not self.length_connection.is_open:
                    self.length_connection = self._create_connection()
                    self.length_channel = self._create_channel(self.length_connection)
                queue = self.length_channel.queue_declare(queue=self.queue_name, passive=True)
                return queue.method.message_count
            except (pika.exceptions.AMQPError, pika.exceptions.StreamLostError) as e:
                print(f"Error getting queue length: {e}")
                # Attempt to recreate the connection and channel
                self.length_connection = self._create_connection()
                self.length_channel = self._create_channel(self.length_connection)
                return 0  # Return 0 if we couldn't get the actual count
