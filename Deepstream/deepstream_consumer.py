import threading
import traceback
from models.message_processing_state import MessageProcessingState
import asyncio
from Shared.models.processing_message import ProcessingMessage
from Shared.services.stores.jobs import BaseJobManager
from Shared.services.stores.media.base_image_storage import BaseImageStorage
from Shared.services.stores.media.base_video_storage import BaseVideoStorage
from Shared.services.queue import RabbitMQQueue,InMemoryProcessingQueue
from perf_data import PerfDataSingleton,PERF_DATA
from queue import Empty as EmptyQueue
import threading
import traceback
from Shared.models.job_status import JobStatus
import asyncio
from queue import Queue
from services.deepstream_pipeline import DeepstreamPipeline

class DeepstreamConsumer:
    def __init__(self,
                 queue: InMemoryProcessingQueue | RabbitMQQueue,
                 job_manager: BaseJobManager,
                 image_storage: BaseImageStorage,
                 video_storage: BaseVideoStorage,
                 pipeline:DeepstreamPipeline,
                 URL: str = ""):
        self.image_storage = image_storage
        self.video_storage = video_storage
        self.URL = URL
        self.queue = queue
        self.job_manager=job_manager
        self.processing_lock = threading.Lock()
        self.pipeline=pipeline
        # Create a queue for passing messages between threads
        self.message_queue = Queue()
        self.in_processing:dict[str,MessageProcessingState]={}# a list of video processing states with delivery tag
        # Store the event loop or create a new one if needed
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            # No event loop in current thread, create a new one
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

    def start_consuming(self):

        perf_data:PERF_DATA=PerfDataSingleton().perf_data

        self.pipeline.start_pipeline(self.handle_completion,asyncio.get_running_loop())
        # Start the consumer thread
        self.thread = threading.Thread(target=self._consumer_thread)
        self.thread.daemon = True
        self.thread.start()
        # Start the async message processor
        self.loop.create_task(self._process_message_queue())

    async def handle_completion(self,input):
        for delivery_tag in self.in_processing:
            if (job:=self.in_processing[delivery_tag]).has_file(str(input.source_path)):
                if job.complete(str(input.source_path)):
                    await self.job_manager.complete_job(job.job_id,job.completed,job.failed)
                    self.queue.complete_task(delivery_tag)
                    del self.in_processing[delivery_tag]

    def _consumer_thread(self):
        """Thread that receives messages from the queue and passes them to the async processor"""
        try:
            self.queue.start_consuming(self._handle_queue_message)
        except Exception as e:
            print(f"Consumer thread error: {str(e)}")
            traceback.print_exc()

    def _handle_queue_message(self, message: ProcessingMessage, delivery_tag):
        """Callback function that puts messages into our internal queue"""
        self.message_queue.put((message, delivery_tag))
        

    async def _process_message_queue(self):
        """Async task that processes messages from our internal queue"""
        while True:
            try:
                # Use executor to avoid blocking the event loop when getting messages
                message, delivery_tag = await self.loop.run_in_executor(
                    None, self.message_queue.get, True, 1.0
                )
                
                await self.job_manager.update_job_status(message.id,job_status=JobStatus.processing)        
                if self.processing_lock.acquire(blocking=False):
                    try:
                        await self.handle_task(message,delivery_tag)
                    except Exception as e:
                        print(f"Error processing task: {str(e)}")
                        traceback.print_exc()
                    finally:
                        self.processing_lock.release()
                        
            except EmptyQueue:
                # Queue timeout, continue polling
                continue
            except Exception as e:
                print(f"Error in message queue processor: {str(e)}")
                traceback.print_exc()
                await asyncio.sleep(1)  # Avoid tight loop on persistent errors

    async def handle_task(self, message: ProcessingMessage,delivery_tag):
        """Handle individual processing tasks"""
        print(f"Processing Videos: {message.data_paths}")
        try:
            saved=[]
            for filename in message.data_paths:
                file_path:str=self.video_storage.download(filename,message.return_detector,False)
                saved.append(file_path)

            self.in_processing[delivery_tag]=MessageProcessingState(saved,message.id)
            self.pipeline.add_sources(saved)
                
        except Exception as e:
            print(f"Task handling error: {str(e)}")
            traceback.print_exc()
            self.in_processing.pop(delivery_tag)
            await self.job_manager.fail_job(message.id, str(e))
