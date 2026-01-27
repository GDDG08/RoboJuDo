from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class DetectionResult(idl.IdlStruct, typename="DetectionModule::DetectionResult"):
    class_id: str
    class_name: str
    box: types.array[types.float32, 4]
    score: types.float32
    xyz: types.array[types.float32, 3]
    offset: types.array[types.float32, 2]
    offset_fov: types.array[types.float32, 2]


@dataclass
@annotate.appendable
@annotate.autoid("sequential")
class DetectionResults(idl.IdlStruct, typename="DetectionModule::DetectionResults"):
    results: types.sequence["robojudo.dds.detection_module.DetectionResult"]
