from tritonclient.http import InferenceServerClient

class TritonClientHandler:
    _instance=None
    def init(server_url):
        TritonClientHandler._instance = InferenceServerClient(url=server_url)
        print("test")
    def infer(model_name:str,inputs):
        if TritonClientHandler._instance is None:
            TritonClientHandler.init("localhost:8080")
        detection_response = TritonClientHandler._instance.infer(model_name=model_name, inputs=inputs)
        return detection_response;