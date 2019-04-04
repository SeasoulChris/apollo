#!/usr/bin/env python

"""Streaming listener class that monitors each of streaming status"""

import pyspark.streaming as streaming

import fueling.common.colored_glog as glog


class DriverStreamingListener(streaming.StreamingListener):
    """Driver streaming listener"""
    @staticmethod
    def onBatchCompleted(batchCompleted):
        """Batch completed"""
        glog.info("DriverListener: Batch Completed: \n\t{}\n".format(batchCompleted))

    @staticmethod
    def onBatchStarted(batchStarted):
        """Batch started"""
        glog.info("DriverListener: Batch Started: \n\t{}\n".format(batchStarted))

    @staticmethod
    def onBatchSubmitted(batchSubmitted):
        """Batch submitted"""
        glog.info("DriverListener: Batch submitted: \n\t{}\n".format(batchSubmitted))

    @staticmethod
    def onOutputOperationCompleted(outputOperationCompleted):
        """Output completed"""
        glog.info("DriverListener: Job of batch has completed: \n\t{}\n".format(
            outputOperationCompleted))

    @staticmethod
    def onOutputOperationStarted(outputOperationStarted):
        """Output started"""
        glog.info("DriverListener: Job of a batch has started: \n\t{}\n".format(
            outputOperationStarted))

    @staticmethod
    def onReceiverError(receiverError):
        """Receiver error"""
        glog.info("DriverListener: Receiver has reported an error: \n\t{}\n".format(receiverError))

    @staticmethod
    def onReceiverStarted(receiverStarted):
        """Receiver started"""
        glog.info("DriverListener: Receiver has been started: \n\t{}\n".format(receiverStarted))

    @staticmethod
    def onReceiverStopped(receiverStopped):
        """Receiver stopped"""
        glog.info("DriverListener: Receiver has stopped: \n\t{}\n".format(receiverStopped))
