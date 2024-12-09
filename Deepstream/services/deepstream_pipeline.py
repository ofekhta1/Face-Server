#!/usr/bin/env python3
import sys
import gi

from services.data_uploader import DataUploader
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from models.file_video_input import FileVideoInput
import pyds
import asyncio
import threading
import sys
import os
from input_handler import InputHandler
import numpy as np
from utils.ctypes import *
from perf_data import PerfDataSingleton,PERF_DATA
import ctypes
from .util import transform_imgs_coords
from models.file_video_input import FileVideoInput
from Shared.models.face import Face
from Shared.models.stored_embedding import FaceEmbedding
from Shared.services.models.triton_model_loader import TritonModelLoader
from Shared.services.util import face_path
from services.pipeline_builder import build_pipeline,create_source_bin
from Shared.services.logging.console_logger import ConsoleLogger

class DeepstreamPipeline:

    def __init__(self, output_settings: dict,data_uploader:DataUploader,logger:ConsoleLogger):
        Gst.init(None)  # Initialize GStreamer
        self.settings = output_settings
        self.perf_data: PERF_DATA = PerfDataSingleton().perf_data
        self.loop = None
        self.pipeline:Gst.Pipeline = None
        self.input_handler = InputHandler()
        self.bus = None
        self.logger=logger
        self.data_uploader=data_uploader
        self.loader=TritonModelLoader()

        self.source_bin_list:dict[int,Gst.Bin]={}



    def bus_call(self,bus, message:Gst.Message, loop):
        t = message.type
        if t == Gst.MessageType.EOS:
            sys.stdout.write("End-of-stream\n")
            self.streammux.set_state(Gst.State.NULL)

            self.pipeline.set_state(Gst.State.READY)
            self.streammux.set_state(Gst.State.READY)
            print("Waiting for new sources")
            # loop.quit()
        elif t==Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            sys.stderr.write("Warning: %s: %s\n" % (err, debug))
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            sys.stderr.write("Error: %s: %s\n" % (err, debug))
            loop.quit()
        elif t == Gst.MessageType.ELEMENT:
            struct = message.get_structure()
            #Check for stream-eos message
            if struct is not None and struct.has_name("stream-eos"):
                parsed, stream_id = struct.get_uint("stream-id")
                if parsed:
                    #Set eos status of stream to True, to be deleted in delete-sources
                    self.handle_eos(stream_id)
                    print("Got EOS from stream %d" % stream_id)
        return True
    
    def handle_eos(self,stream_id):
        input=self.input_handler.get_source(stream_id)

        self.data_uploader.upload_processed_input_data(input)
        self.remove_source(stream_id)
        if self.eos_callback:
            asyncio.run_coroutine_threadsafe(self.eos_callback(input), self.consumer_loop)

            
    def start_pipeline(self,eos_callback=None,consumer_loop=None):
        """Start the GStreamer pipeline."""

        self.pipeline,self.streammux = build_pipeline(self.settings,self.probe_func)
        self.eos_callback=eos_callback
        self.consumer_loop:asyncio.BaseEventLoop=consumer_loop
        # Create an event loop and attach the bus to it
        self.loop = GLib.MainLoop()
        self.bus = self.pipeline.get_bus()
        self.bus.add_signal_watch()
        self.bus.connect("message", self.bus_call, self.loop)

        # Add a callback to print performance data every 5 seconds
        GLib.timeout_add(5000, self.perf_data.perf_print_callback)

        # Start playback
        self.logger.info("Starting pipeline with sources:")

        self.pipeline.set_state(Gst.State.PLAYING)
        # Run the GStreamer loop in a separate thread
        def run_loop():
            try:
                self.loop.run()
            except Exception as ex:
                self.logger.error(f"Error occurred in GStreamer loop: {ex}")
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()

        self.logger.info("GStreamer loop started in a separate thread.")

    def stop_pipeline(self):
        """Stop the GStreamer pipeline."""
        if self.pipeline:
            self.logger.info("Stopping pipeline...")
            self.pipeline.set_state(Gst.State.NULL)
            self.loop.quit() if self.loop else None
            self.logger.info("Pipeline stopped.")

    def get_available_stream_id(self):
        """
        Finds the smallest integer key >= 0 that is not in the dictionary.
        :param d: Dictionary with integer keys.
        :return: The first missing key >= 0.
        """
        key = 0
        while key in self.source_bin_list:
            key += 1
        return key

    def add_sources(self,sources:list[str]):
        perf_data=PerfDataSingleton().perf_data
        folder_name="/tmp"

        for i in range(len(sources)):
            try:
                os.rmdir(folder_name + "/stream_" + str(i))
            except:
                pass

            os.mkdir(folder_name + "/stream_" + str(i))

            source_id=self.get_available_stream_id()
            print("Creating source_bin ", source_id, " \n ")
            uri_name = sources[i]
            if uri_name.find("rtsp://") == 0:
                is_live = True
            else:
                input:FileVideoInput=InputHandler.process_input_video(uri_name,(self.settings["Height"],self.settings["Width"]));
                if input is None:
                    self.logger.error(f"Failed to process the input: {uri_name}")
                self.input_handler.register_source(input,source_id)
            source_bin = create_source_bin(input.uri,source_id)
            self.pipeline.add(source_bin)
            padname="sink_%u" %source_id
            sinkpad= self.streammux.request_pad_simple(padname) 
            if not sinkpad:
                sys.stderr.write("Unable to create sink pad bin \n")
            srcpad=source_bin.get_static_pad("src")
            if not srcpad:
                sys.stderr.write("Unable to create src pad bin \n")
            srcpad.link(sinkpad)
            state_return = source_bin.set_state(Gst.State.PLAYING)
            if(len(self.source_bin_list.keys())==0):
                self.pipeline.set_state(Gst.State.PLAYING)

            if state_return == Gst.StateChangeReturn.SUCCESS:
                print("STATE CHANGE SUCCESS\n")
                perf_data.add_stream(f"stream-{source_id}")
                self.source_bin_list[source_id]=source_bin

                

            elif state_return == Gst.StateChangeReturn.FAILURE:
                print("STATE CHANGE FAILURE\n")
            
            elif state_return == Gst.StateChangeReturn.ASYNC:
                state_return = source_bin.get_state(Gst.CLOCK_TIME_NONE)
                perf_data.add_stream(f"stream-{source_id}")
                self.source_bin_list[source_id]=source_bin


            elif state_return == Gst.StateChangeReturn.NO_PREROLL:
                print("STATE CHANGE NO PREROLL\n")
        
    def remove_source(self, source_id: str):
        """Dynamically remove a source from the pipeline."""
        if not self.pipeline:
            raise RuntimeError("Pipeline is not running.")
        self.logger.info(f"Removing source: {source_id}")

        source_bin=self.source_bin_list[source_id]
        #Attempt to change status of source to be released 
        state_return = source_bin.set_state(Gst.State.NULL)
        for child in source_bin.children:
            child.set_state(Gst.State.NULL)
            source_bin.remove(child)
        srcpad=source_bin.get_static_pad("src")
        srcpad.send_event(Gst.Event.new_flush_stop(True))
        source_bin.release_request_pad(srcpad)
        self.input_handler.remove_source(source_id)

        if state_return == Gst.StateChangeReturn.SUCCESS:
            print("STATE CHANGE SUCCESS\n")
            pad_name = "sink_%u" % source_id
            print(pad_name)
            #Retrieve sink pad to be released
            sinkpad = self.streammux.get_static_pad(pad_name)
            #Send flush stop event to the sink pad, then release from the streammux
            sinkpad.send_event(Gst.Event.new_flush_stop(True))
            self.streammux.release_request_pad(sinkpad)
            print("STATE CHANGE SUCCESS\n")
            #Remove the source bin from the pipeline
            self.perf_data.remove_stream(f"stream-{source_id}")
            self.pipeline.remove(source_bin)
            del self.source_bin_list[source_id]

        elif state_return == Gst.StateChangeReturn.FAILURE:
            print("STATE CHANGE FAILURE\n")
        
        elif state_return == Gst.StateChangeReturn.ASYNC:
            state_return = source_bin.get_state(Gst.CLOCK_TIME_NONE)
            pad_name = "sink_%u" % source_id
            sinkpad = self.streammux.get_static_pad(pad_name)
            sinkpad.send_event(Gst.Event.new_flush_stop(False))
            self.streammux.release_request_pad(sinkpad)
            print("STATE CHANGE ASYNC\n")
            self.perf_data.remove_stream(f"stream-{source_id}")
            self.pipeline.remove(source_bin)
            del self.source_bin_list[source_id]



    def probe_func(self,pad, info):
        perf_data=PerfDataSingleton().perf_data
        gst_buffer = info.get_buffer()

        if not gst_buffer:
            print("Unable to get GstBuffer ")
            return

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
                # The casting is done by pyds.NvDsFrameMeta.cast()
                # The casting also keeps ownership of the underlying memory
                # in the C code, so the Python garbage collector will leave
                # it alone.
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                frame_number = frame_meta.frame_num
                
                input:FileVideoInput=self.input_handler.get_source(frame_meta.source_id);
                t_frame=frame_meta.frame_user_meta_list

                user_meta=pyds.NvDsUserMeta.cast(t_frame.data)
                
                meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)  # Cast user_meta to NvDsInferTensorMeta
                outputs={}
                for i in range(meta.num_output_layers):
                    layer = pyds.get_nvds_LayerInfo(meta, i)
                    outputs[layer.layerName]=layer


                for detector_name in self.loader.model_registry["detectors"]:
                    
                    faces_len_layer=outputs[f"{detector_name}_dets_len"]
                    faces_len = convert_to_num(faces_len_layer.buffer,ctypes.c_uint8)
                    
                    genders_layer=outputs[f"{detector_name}_genders"]
                    ages_layer=outputs[f"{detector_name}_ages"]
                    quality_layer=outputs[f"{detector_name}_quality_scores"]
                    kpss_layer=outputs[f"{detector_name}_kps"]
                    dets_layer=outputs[f"{detector_name}_dets"]

                    genders=["M" if gender ==1 else "W" for gender in convert_to_array(genders_layer.buffer,(faces_len,1),ctypes.c_int32)]
                    ages=[age[0] for age in convert_to_array(ages_layer.buffer,(faces_len,1),ctypes.c_int32)]

                    qualities=convert_to_array(quality_layer.buffer,(faces_len,1))
                    kpss = convert_to_array(kpss_layer.buffer,(faces_len,5,2))
                    dets = convert_to_array(dets_layer.buffer,(faces_len,5))
                    if len(dets)==0:
                        continue
                    dets,kpss=transform_imgs_coords(dets,kpss,input.scale,input.paddings)

                    faces = [
                        Face(bbox=det[0:4], kps=kps, det_score=det[4], quality=quality[0],gender=gender,age=age)
                        for det, kps, quality,gender,age in zip(dets, kpss, qualities,genders,ages)
                    ]

                    sorted_indices = sorted(
                    range(len(faces)), key=lambda x: faces[x]["quality"], reverse=True
                    )
                    faces = [faces[i] for i in sorted_indices]

                    for embedder_name in self.loader.model_registry["embedders"]:
                        face_embeddings: list[FaceEmbedding] = []
                        layer=outputs[f"{detector_name}_{embedder_name}_embeddings"]
                        embeddings=convert_to_array(layer.buffer,(faces_len,512))

                        embeddings = [
                            embeddings[i] for i in sorted_indices
                        ]  # sort by quality,so it matches the faces order

                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

                        # Normalize each row
                        normalized_embeddings = embeddings / norms
                        for i, face in enumerate(faces):
                            aligned_image_name = face_path(f"{input.path.stem}_{frame_number}.png",i)
                            bbox = [int(coord) for coord in face["bbox"]]
                            landmarks = [(x[0], x[1]) for x in face["kps"]]
                            quality = face["quality"] if "quality" in face else 1
                            age = face["age"] if "age" in face else -1
                            gender = face["gender"] if "gender" in face else ""
                            fe = FaceEmbedding(
                                aligned_image_name,
                                bbox,
                                normalized_embeddings[i],
                                quality=quality,
                                landmarks=landmarks,
                                age=age,
                                gender=gender,
                            )
                            face_embeddings.append(fe)

                        input.emb_manager.add_embedding_typed(
                        face_embeddings, detector_name, embedder_name
                        )
                pts = frame_meta.buf_pts
                stream_index = "stream-{0}".format(frame_meta.pad_index)
                perf_data.update_fps(stream_index)
            except StopIteration:
                break
            except Exception as ex:
                breakpoint()
                print(ex)
            l_frame = l_frame.next

        return Gst.PadProbeReturn.OK