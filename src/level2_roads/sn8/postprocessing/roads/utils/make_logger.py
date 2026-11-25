# Copied from https://github.com/SpaceNetChallenge/SpaceNet8 (Apache-2.0),
# commit 30ca30c2530c6a234d040f60a99a7366ff62d756, file baseline/postprocessing/roads/utils/make_logger.py.
# Changes: none.
# Copyright of the original: the SpaceNet 8 authors; licence: third_party/spacenet8/LICENSE.
# ruff: noqa
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:42:15 2019

@author: avanetten
"""

import logging

# include errors?
# https://docs.python.org/3/howto/logging-cookbook.html#buffering-logging-messages-and-outputting-them-conditionally

loggers = {}

###############################################################################
def make_logger(log_file, logger_name='log', write_to_console=True):
    # https://docs.python.org/3/howto/logging-cookbook.html#logging-to-multiple-destinations
    # set up logging to file - see previous section for more details
    global loggers
    
    name_tmp = ''
    if loggers.get(name_tmp):
        return  logging.StreamHandler(), loggers.get(name_tmp)
    
    logging.basicConfig(level=logging.INFO,
    #logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-8s %(levelname)-8s %(message)s',
                        #format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    if write_to_console:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-8s: %(levelname)-8s %(message)s')
        #formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)    

    else:
        console = ''
        

    # add the handler to the root logger
    # z = logging.getLogger('')
    # # print(z)
    # if z.hasHandlers():
    #     print("handlers present")
    
    #ll = logging.getLogger('')
    #if not ll.handlers:
    #    ll.addHandler(console)

    logger0 = logging.getLogger(logger_name)
    
    # # clear multiple loggers
    # if (logger0.hasHandlers()):
    #     print("logger0 handlsers:")
    #     logger0.handlers.clear()
    #     logging.getLogger('').addHandler(console)

    return console, logger0
      
