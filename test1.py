#!/usr/bin/env python3

################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2019-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

import pyds
from gi.repository import GLib, Gst
import gi
import os
import sys
from draw_image import draw_on_image

gi.require_version('Gst', '1.0')

import numpy as np
from copy import deepcopy


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

live_stream = True
use_v4 = False  # False to use v7


def blockPrint():
    sys.stdout = open(os.devnull, 'w')
    
def enablePrint():
    sys.stdout = sys.__stdout__

def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s\n" % (err, debug))
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s\n" % (err, debug))
        loop.quit()
    return True


def add_obj_meta_to_frame(batch_meta, frame_meta, box):
    """ Inserts an object into the metadata """
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params
    rect_params.left = box['left']
    rect_params.top = box['top']
    rect_params.width = box['right'] - box['left']
    rect_params.height = box['bottom'] - box['top']

    # Semi-transparent yellow backgroud
    rect_params.has_bg_color = 0
    rect_params.bg_color.set(1, 1, 0, 0.4)

    # Red border of width 3
    rect_params.border_width = 3
    rect_params.border_color.set(1, 0, 0, 1)

    # Set object info including class, detection confidence, etc.
    obj_meta.confidence = 1.0
    obj_meta.class_id = 6

    # There is no tracking ID upon detection. The tracker will
    # assign an ID.
    obj_meta.object_id = 1

    # lbl_id = 1
    # if lbl_id >= len(label_names):
    #     lbl_id = 0

    # # Set the object classification label.
    # obj_meta.obj_label = label_names[lbl_id]

    # # Set display text for the object.
    # txt_params = obj_meta.text_params
    # if txt_params.display_text:
    #     pyds.free_buffer(txt_params.display_text)

    # txt_params.x_offset = int(rect_params.left)
    # txt_params.y_offset = max(0, int(rect_params.top) - 10)
    # txt_params.display_text = (
    #     label_names[lbl_id] + " " + "{:04.3f}".format(frame_object.detectionConfidence)
    # )
    # # Font , font-color and font-size
    # txt_params.font_params.font_name = "Serif"
    # txt_params.font_params.font_size = 10
    # # set(red, green, blue, alpha); set to White
    # txt_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

    # # Text background color
    # txt_params.set_bg_clr = 1
    # # set(red, green, blue, alpha); set to Black
    # txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

    # # Inser the object into current frame meta
    # # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)


def draw_placeholder(batch_meta, frame_meta, bbox):
    if len(bbox)==0: return
    for box in bbox:
        add_obj_meta_to_frame(batch_meta, frame_meta, box)
    

def osd_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    l_frame = batch_meta.frame_meta_list
    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), 0)
    rgba = np.array(n_frame, copy=True, order='C')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]
    rgb = np.zeros((720,1280,3), dtype='uint8')
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    # print(frame_copy[0][0][0])
    frame_meta_ = None
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            frame_meta_ = frame_meta
        except StopIteration:
            break
            
        # Intiallizing object counter with 0.
        frame_number = frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj = frame_meta.obj_meta_list

        list_bbox = []
        temp_dict = {'Left':0, 'Top': 0, 'Right': 0, 'Bottom': 0, 'Conf': 0.0, 'ObjectClassName': '1'}
        label_list = ['rack_1','rack_2','rack_3','rack_4','klt_box_empty','klt_box_full']
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break

            bbox = deepcopy(temp_dict)
            bbox['Left'] = obj_meta.rect_params.left
            bbox['Top'] = obj_meta.rect_params.top
            bbox['Right'] = bbox['Left'] + obj_meta.rect_params.width
            bbox['Bottom'] = bbox['Top'] + obj_meta.rect_params.height
            bbox['Conf'] = round(obj_meta.confidence, 2)
            bbox['ObjectClassName'] = label_list[obj_meta.class_id]

            list_bbox.append(bbox)

            cls_id = obj_meta.class_id
            obj_meta.text_params.display_text = ("")
            obj_meta.text_params.font_params.font_color.set(1.0,1.0,1.0,0.0)
            obj_meta.text_params.text_bg_clr.set(0.0,0.0,0.0,0.0)
            
            if cls_id == 0: obj_meta.rect_params.border_color.set(0.0, 0.0, 1.0, 0.8)
            elif cls_id == 1: obj_meta.rect_params.border_color.set(1.0, 1.0, 0.0, 0.8)
            elif cls_id == 2: obj_meta.rect_params.border_color.set(1.0, 0.64, 0.0, 0.8)
            elif cls_id == 3: obj_meta.rect_params.border_color.set(0.9, 0.9, 1.0, 0.8)
            elif cls_id == 4: obj_meta.rect_params.border_color.set(0.5, 0.0, 0.5, 0.8)
            else: obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 0.8)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        
        
        # infer place holder 
        blockPrint()
        available_Pholders = draw_on_image(rgb, list_bbox)
        draw_placeholder(batch_meta, frame_meta_, available_Pholders)
        enablePrint()


        # # Acquiring a display meta object. The memory ownership remains in
        # # the C code so downstream plugins can still access it. Otherwise
        # # the garbage collector will claim it when this probe function exits.
        # display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        # display_meta.num_labels = 1
        # py_nvosd_text_params = display_meta.text_params[0]
        # # Setting display text to be shown on screen
        # # Note that the pyds module allocates a buffer for the string, and the
        # # memory will not be claimed by the garbage collector.
        # # Reading the display_text field here will return the C address of the
        # # allocated string. Use pyds.get_string() to get the string content.
        # py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={}".format(
        #     frame_number, num_rects)

        # # Now set the offsets where the string should appear
        # py_nvosd_text_params.x_offset = 10
        # py_nvosd_text_params.y_offset = 12

        # # Font , font-color and font-size
        # py_nvosd_text_params.font_params.font_name = "Serif"
        # py_nvosd_text_params.font_params.font_size = 10
        # # set(red, green, blue, alpha); set to White
        # py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # # Text background color
        # py_nvosd_text_params.set_bg_clr = 1
        # # set(red, green, blue, alpha); set to Black
        # py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    if not caps:
        caps = decoder_src_pad.query_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio.
    print("gstname=", gstname)
    if (gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        print("features=", features)
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write(
                    "Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(
                " Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if (name.find("decodebin") != -1):
        Object.connect("child-added", decodebin_child_added, user_data)

    if "source" in name:
        source_element = child_proxy.get_by_name("source")
        if source_element.find_property('drop-on-latency') != None:
            Object.set_property("drop-on-latency", True)


file_loop = False
def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    if file_loop:
        # use nvurisrcbin to enable file-loop
        uri_decode_bin = Gst.ElementFactory.make(
            "nvurisrcbin", "uri-decode-bin")
        uri_decode_bin.set_property("file-loop", 1)
        uri_decode_bin.set_property("cudadec-memtype", 0)
    else:
        uri_decode_bin = Gst.ElementFactory.make(
            "uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(
        Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    pipeline = Gst.Pipeline()
    # print(type(pipeline))

    nvdslogger = Gst.ElementFactory.make("nvdslogger", "logger")
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

    src_file = 'file://'+ os.getcwd() + '/eval_video_1.mp4'
    # Source element for reading from the file

    source = create_source_bin(0, src_file)

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")

    streammux.set_property('gpu-id', 0)
    streammux.set_property('live-source', 0)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)
    streammux.set_property('width', IMAGE_WIDTH)
    streammux.set_property('height', IMAGE_HEIGHT)
    streammux.set_property('enable-padding', 0)
    streammux.set_property('nvbuf-memory-type', mem_type)

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")

    if use_v4:
        pgie.set_property('config-file-path', "config_infer_primary_yoloV4.txt")
    else:
        pgie.set_property('config-file-path', "config_infer_primary_yoloV7.txt")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvvidconv.set_property("nvbuf-memory-type", mem_type)

    # # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvosd.set_property("display-text", 1)
    nvosd.set_property("process-mode", 1)    # 0 CPU, 1 GPU

    # Finally render the osd output
    # sink = Gst.ElementFactory.make("fakesink", "fakesink")
    if live_stream:
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    else:
        sink = Gst.ElementFactory.make("filesink", "filesink")
        sink.set_property("location", "output.mp4")
        sink.set_property("sync", 0)
        sink.set_property("async", 0)
    
    # save file
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)
    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    encoder.set_property("bitrate", 2000000)
    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    container = Gst.ElementFactory.make("qtmux", "qtmux")


    print("Adding elements to Pipeline \n")
    pipeline.add(nvdslogger)
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)

    # we link the elements together
    print("Linking elements in the Pipeline \n")
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad = source.get_static_pad("src")

    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvdslogger)
    nvdslogger.link(nvvidconv)
    nvvidconv.link(nvosd)


    if live_stream:
        nvosd.link(sink)
    else:
        nvosd.link(nvvidconv2)
        nvvidconv2.link(capsfilter)
        capsfilter.link(encoder)
        encoder.link(codeparser)
        codeparser.link(container)
        container.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))