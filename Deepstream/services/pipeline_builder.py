import gi
import sys

import gi.repository
gi.require_version("Gst", "1.0")
sys.path.append('../../')
sys.path.append('/common')
from gi.repository import Gst ,GLib # type: ignore
import sys;
import pyds
import os
from collections.abc import Callable
from utils.ctypes import *
from Shared.services.models.triton_model_loader import TritonModelLoader

global loader
loader=TritonModelLoader()

def configure_streammux(streammux,settings):

    streammux.set_property('width', settings["Width"] )
    streammux.set_property('height', settings["Height"])
    streammux.set_property('batch-size', 4)
    streammux.set_property('batched-push-timeout', 4000000)
    streammux.set_property('sync-inputs',0)
    streammux.set_property('enable-padding',1)
    streammux.set_property('live-source',0)
    return streammux;


def create_source_bin(uri,stream_id):
    print("Creating Source \n ")
    
    bin_name=f"source-bin-{stream_id}"

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    print(bin_name)
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")

    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,stream_id)
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if gstname.find("video") != -1:
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            #Get a sink pad from the streammux, link to decodebin
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
            
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")




def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if name.find("decodebin") != -1:
        Object.connect("child-added", decodebin_child_added, user_data)


def build_pipeline(settings:dict,probe_func:Callable):

    print("Creating Pipeline \n ")
    pipeline:Gst.Pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating streamux \n ")

    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
        return None;

    streammux= configure_streammux(streammux,settings)
    print("Creating VideoRate \n ")
    videorate = Gst.ElementFactory.make("videorate", "video-rate")
    if not videorate:
        sys.stderr.write(" Unable to create videorate limiter \n")
        return pipeline;

    videorate.set_property("drop-only",True)
    videorate.set_property("max-rate",settings["FPS"])

    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinferserver", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create InferServer \n")
        return pipeline;
    
    pgie.set_property('config-file-path', os.path.join("config","pgie.pbtxt"))
    # pgie_batch_size = pgie.get_property("batch-size")
    # if (pgie_batch_size != number_sources):
    #     # print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ",
    #         #   number_sources, " \n")
    #     pgie.set_property("batch-size", number_sources)

    
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
        return pipeline;

    print("Creating filter1\n ")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
        return pipeline;
    filter1.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    sink=Gst.ElementFactory.make("fakesink","fakesink")

    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    streammux.set_property("nvbuf-memory-type", mem_type)
    nvvidconv1.set_property("nvbuf-memory-type", mem_type)

    pipeline.add(streammux)
    pipeline.add(videorate)
    pipeline.add(pgie)
    pipeline.add(nvvidconv1)
    pipeline.add(filter1)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(videorate) 
    videorate.link(pgie)  
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(sink)
    filter1_sink_pad = filter1.get_static_pad("sink")
    if not filter1_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        filter1_sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_func)
    return pipeline,streammux