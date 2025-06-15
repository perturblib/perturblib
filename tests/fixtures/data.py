"""Copyright (C) 2025  GlaxoSmithKline plc

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

import pytest

import perturb_lib as plib


@pytest.fixture(scope="module")
def in_memory_plib_data():
    context = ["DummyData", "DummyDataLongStrings"]
    return plib.load_plibdata(context, plibdata_type=plib.InMemoryPlibData)


@pytest.fixture(scope="module")
def on_disk_plib_data():
    context = ["DummyData", "DummyDataLongStrings"]
    return plib.load_plibdata(context, plibdata_type=plib.OnDiskPlibData)


@pytest.fixture(scope="module")
def list_of_plibdata():
    context = ["DummyData", "DummyDataLongStrings"]
    inmem_data = plib.load_plibdata(context, plibdata_type=plib.InMemoryPlibData)
    ondisk_data = plib.load_plibdata(context, plibdata_type=plib.OnDiskPlibData)
    return [inmem_data, ondisk_data]
