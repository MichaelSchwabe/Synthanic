  *	    ?h@2j
3Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map???V?/??!????K@)HP?s??1?S?r
^7@:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2
ףp=
??!~h???6@)?D???J??1????>4)@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice Dio??ɔ?!n۶m۶$@)Dio??ɔ?1n۶m۶$@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV22??%䃞?!?Cc}h.@)??ǘ????1??>4և @:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2?sF????!??X?.@)ŏ1w-!??1??)x9@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice ?W[?????!4և???@)?W[?????14և???@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice y?&1???!$I?$I?@)y?&1???1$I?$I?@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice _?Qڋ?!?r
^N?@)_?Qڋ?1?r
^N?@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2S?!?uq??!և???X+@)-C??6??1????X@:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip???3???!Dc}h??D@)?g??s?u?1i????@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map+??ݓ???!????>tE@)??_vOf?1??S?r
??:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Zip???QI???!*x9/?X@)????Mb`?1????S??:Preprocessing2F
Iterator::ModelJ+???!      Y@)ŏ1w-!_?1??)x9??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip?ʡE????!ׇ????=@)ŏ1w-!_?1??)x9??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism*??D???!i????X@)??H?}]?1Z?Cc??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map?W[?????!4և???>@)??_?LU?1N??)x9??:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTakef??a?ִ?!?X??D@)-C??6J?1????X??:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake?c?ZB??!*x9/'>@)a2U0*?C?1???????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.