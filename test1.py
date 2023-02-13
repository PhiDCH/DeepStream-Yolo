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

from copy import deepcopy
import numpy as np
import pyds
from gi.repository import GLib, Gst
import gi
import os
import glob
import sys
from draw_image import draw_on_image

from FPS import PERF_DATA

gi.require_version('Gst', '1.0')


IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1280

live_stream = False
use_v4 = False  # False to use v7

perf_data = None


def blockPrint():
    sys.stdout = open(os.devnull, 'w')


def enablePrint():
    sys.stdout = sys.__stdout__


output_txt_per_frame = ''
output_txt = []

def dump_output_per_frame(obj_meta):
    global output_txt_per_frame
    cls_id, x, y, w, h, conf = obj_meta.class_id, obj_meta.rect_params.left, obj_meta.rect_params.top, obj_meta.rect_params.width, obj_meta.rect_params.height, obj_meta.confidence
    txt = voc2yolo(cls_id, x, y, w, h, conf)
    if output_txt_per_frame:
        output_txt_per_frame += '\n' + txt
    else:
        output_txt_per_frame += txt

def clean_folder(folder_path):
    files = glob.glob(f'{folder_path}/*')
    for f in files:
        os.remove(f)

def dump_out():
    clean_folder('output/')
    for i, line in enumerate(output_txt):
        video_name = 'eval_video_1.mp4'
        save_name = video_name.replace('.mp4', f'_{i + 1}.txt')
        with open(f'output/{save_name}', 'w+') as f:
            f.write(line)
        # with open('output/{}.txt'.format(str(i)), 'w+') as f:
        #     f.write(line)

def voc2yolo(cls_id, x, y, w, h, conf, W=IMAGE_WIDTH, H=IMAGE_HEIGHT):
    x_center, y_cent = x + w/2, y + h/2
    w, h = w/W, h/H
    x_center, y_cent = x_center/W, y_cent/H
    txt = ' '.join(str(i) for i in [cls_id, x_center, y_cent, w, h, conf])
    return txt


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


def add_obj_meta_to_frame(batch_meta, frame_meta, box, text=False):
    """ Inserts an object into the metadata """
    # this is a good place to insert objects into the metadata.
    # Here's an example of inserting a single object.
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    # Set bbox properties. These are in input resolution.
    rect_params = obj_meta.rect_params
    if text:
        rect_params.left = box['left']
        rect_params.top = box['top']
        rect_params.width = 220
        rect_params.height = 50
        # Semi-transparent yellow backgroud
        rect_params.has_bg_color = 1
        rect_params.bg_color.set(1, 1, 1, 1.0)

        # Red border of width 3
        rect_params.border_width = 0
        rect_params.border_color.set(1, 0, 0, 0.0)
    else:
        rect_params.left = box['left']
        rect_params.top = box['top']
        rect_params.width = box['right'] - box['left']
        rect_params.height = box['bottom'] - box['top']

        # Semi-transparent yellow backgroud
        rect_params.has_bg_color = 0
        rect_params.bg_color.set(1, 1, 0, 0.8)

        # Red border of width 3
        rect_params.border_width = 3
        rect_params.border_color.set(1, 0, 0, 1)

    # Set object info including class, detection confidence, etc.
    obj_meta.confidence = 1.0
    obj_meta.class_id = 6

    if text:
        try:
            fps = int(perf_data.perf_dict['stream0'])
        except:
            fps = 0
        dis_txt = "FPS: {}\t\t\tN_full_KLT: {}\nrack_conf: {}\tN_empty_KLT: {}\n\t\t\t\tN_Pholders: {}".format(
            fps, box['n_full'], box['conf'], box['n_empty'], box['n_pholders'])

        # Set display text for the object.
        txt_params = obj_meta.text_params
        if txt_params.display_text:
            pyds.free_buffer(txt_params.display_text)

        txt_params.x_offset = int(rect_params.left) + 10
        txt_params.y_offset = max(0, int(rect_params.top) + 10)
        txt_params.display_text = dis_txt
        # Font , font-color and font-size
        txt_params.font_params.font_name = "Serif"
        txt_params.font_params.font_size = 8
        # set(red, green, blue, alpha); set to White
        txt_params.font_params.font_color.set(0.0, 0.0, 0.0, 1.0)

        # Text background color
        txt_params.set_bg_clr = 0
        # set(red, green, blue, alpha); set to Black
        txt_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)

        # Inser the object into current frame meta
        # This object has no parent
    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)


def draw_placeholder(batch_meta, frame_meta, bbox):
    if len(bbox) == 0:
        return
    for box in bbox:
        add_obj_meta_to_frame(batch_meta, frame_meta, box)


def draw_board(batch_meta, frame_meta, rack_meta):
    for rack in rack_meta:
        add_obj_meta_to_frame(batch_meta, frame_meta, rack, True)
    return


def osd_sink_pad_buffer_probe(pad, info, u_data):
    global perf_data, output_txt_per_frame, output_txt

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
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]
    rgb = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype='uint8')
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    # print(frame_copy[0][0][0])
    frame_meta_ = None
    while l_frame is not None:
        perf_data.update_fps('stream0')
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
        l_obj = frame_meta.obj_meta_list

        list_bbox = []
        temp_dict = {'Left': 0, 'Top': 0, 'Right': 0,
                     'Bottom': 0, 'Conf': 0.0, 'ObjectClassName': '1'}
        label_list = ['rack_1', 'rack_2', 'rack_3',
                      'rack_4', 'klt_box_empty', 'klt_box_full']
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            
            dump_output_per_frame(obj_meta)

            bbox = deepcopy(temp_dict)
            bbox['Left'] = int(obj_meta.rect_params.left)
            bbox['Top'] = int(obj_meta.rect_params.top)
            bbox['Right'] = int(bbox['Left'] + obj_meta.rect_params.width)
            bbox['Bottom'] = int(bbox['Top'] + obj_meta.rect_params.height)
            bbox['Conf'] = round(obj_meta.confidence, 2)
            bbox['ObjectClassName'] = label_list[obj_meta.class_id]

            list_bbox.append(bbox)

            cls_id = obj_meta.class_id
            obj_meta.text_params.display_text = ("")
            obj_meta.text_params.font_params.font_color.set(1.0, 1.0, 1.0, 0.0)
            obj_meta.text_params.text_bg_clr.set(0.0, 0.0, 0.0, 0.0)

            if cls_id == 0:
                obj_meta.rect_params.border_color.set(0.0, 0.0, 0.0, 1.0)
            elif cls_id == 1:
                obj_meta.rect_params.border_color.set(1.0, 1.0, 0.0, 1.0)
            elif cls_id == 2:
                obj_meta.rect_params.border_color.set(1.0, 0.64, 0.0, 1.0)
            elif cls_id == 3:
                obj_meta.rect_params.border_color.set(0.9, 0.9, 1.0, 1.0)
            elif cls_id == 4:
                obj_meta.rect_params.border_color.set(0.5, 0.0, 0.5, 1.0)
            else:
                obj_meta.rect_params.border_color.set(0.0, 1.0, 0.0, 1.0)
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        blockPrint()
        available_Pholders, list_racks = draw_on_image(rgb, list_bbox)
        enablePrint()
        # draw place holder
        draw_placeholder(batch_meta, frame_meta_, available_Pholders)
        # draw board
        draw_board(batch_meta, frame_meta_, list_racks)
        

        output_txt.append(output_txt_per_frame)
        try:
            l_frame = l_frame.next
            output_txt_per_frame = ''
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
    global perf_data
    perf_data = PERF_DATA(1)
    # Standard GStreamer initialization
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    pipeline = Gst.Pipeline()
    # print(type(pipeline))

    nvdslogger = Gst.ElementFactory.make("nvdslogger", "logger")
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

    src_file = 'file://' + os.getcwd() + '/eval_video_1.mp4'
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
        pgie.set_property('config-file-path',
                          "config_infer_primary_yoloV4.txt")
    else:
        pgie.set_property('config-file-path',
                          "config_infer_primary_yoloV7.txt")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvvidconv.set_property("nvbuf-memory-type", mem_type)

    # # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvosd.set_property("display-text", 1)
    nvosd.set_property("process-mode", 1)    # 0 CPU, 1 GPU

    # Finally render the osd output
    if live_stream:
        # sink = Gst.ElementFactory.make("fakesink", "fakesink")
        sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
        # sink.set_property("qos",0)
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

    queue1 = Gst.ElementFactory.make("queue", "queue1")
    queue2 = Gst.ElementFactory.make("queue", "queue2")
    queue3 = Gst.ElementFactory.make("queue", "queue3")
    queue4 = Gst.ElementFactory.make("queue", "queue4")
    queue5 = Gst.ElementFactory.make("queue", "queue5")
    queue6 = Gst.ElementFactory.make("queue", "queue6")
    pipeline.add(queue1)
    pipeline.add(queue2)
    pipeline.add(queue3)
    pipeline.add(queue4)
    pipeline.add(queue5)
    pipeline.add(queue6)

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
    streammux.link(queue1)
    queue1.link(pgie)
    pgie.link(queue2)
    queue2.link(nvdslogger)
    nvdslogger.link(queue3)
    queue3.link(nvvidconv)
    nvvidconv.link(queue4)
    queue4.link(nvosd)

    if live_stream:
        nvosd.link(queue5)
        queue5.link(sink)
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

    # perf callback function to print fps every 5 sec
    GLib.timeout_add(5000, perf_data.perf_print_callback)

    # start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    pipeline.set_state(Gst.State.NULL)
    dump_out()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
