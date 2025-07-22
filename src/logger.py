"""
Copyright (c) 2025, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
Written by Ben Zhu, Harsh Bhatia, Menglong Zhao, and Xueqiao Xu.
LLNL-CODE-2007259.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging


# ------------------------------------------------------------------------------
def init_logger(level):

    LOG_FMT = '%(asctime)s - %(name)s:%(funcName)s:%(lineno)s - %(levelname)s - %(message)s'

    logger = logging.getLogger()
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)
    sh.setFormatter(logging.Formatter(LOG_FMT))
    logger.addHandler(sh)


# ------------------------------------------------------------------------------
def print_dict(_, indent=0):
    assert isinstance(_, dict)
    assert isinstance(indent, int)
    istr = '   '.join(['' for _ in range(indent)])
    for k,v in _.items():
        if isinstance(v, dict):
            print(f'{istr}[{k}] ==>')
            print_dict(v, indent+2)
        else:
            print(f'{istr}[{k}] ==> [{v}]')

# ------------------------------------------------------------------------------
