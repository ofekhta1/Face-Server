from queue import Queue
class InMemoryProcessingQueue:
    def __init__(self):
        self.queue = Queue()


    def enqueue(self, message):
        if isinstance(message, list):
            for msg in message:
                self.queue.put(msg)
        else:
            self.queue.put(message)
        
    def wait_for_task(self):
        return self.queue.get()
        
    def complete_task(self):
        self.queue.task_done()


        