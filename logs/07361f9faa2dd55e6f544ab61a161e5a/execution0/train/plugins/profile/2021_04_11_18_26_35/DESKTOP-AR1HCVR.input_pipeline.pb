  *	33333Sh@2j
3Iterator::Model::MaxIntraOpParallelism::Zip[1]::Maph"lxz???!??v1T?L@)'???????1k`?k?5@:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2/?$???!`?.?5@)
ףp=
??1??D??'@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice w-!?l??!z??(??&@)w-!?l??1z??(??&@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice +??????!4?.?
$@)+??????14?.?
$@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2????o??!x?gI?/@)?St$????1?Wz??!@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice ???Q???!e?[4?@)???Q???1e?[4?@:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2?X?? ??!P? ?.@)???QI??1<?8??d@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice ?!??u???!0m????@)?!??u???10m????@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2??W?2ġ?!v;????1@)-C??6??1??uO@:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip?O??n??![>_?N$C@)-C??6j?1??uO??:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Zip??ܵ?|??!?B
=x?X@)/n??b?1?1J~???:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::ZipH?z?G??!??sm?WA@)?J?4a?1????$D??:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map?N@aó?!.??D??C@)ŏ1w-!_?1p??[>??:Preprocessing2F
Iterator::ModelԚ?????!      Y@)??H?}]?1B#b?c???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism=?U?????!?;?8??X@)?~j?t?X?1?r|?}???:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map::MapTR'?????!???A?A@)a2U0*?S?1+ϗ???:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTakef?c]?F??!aЈ??XC@)-C??6J?1??uO??:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake??e??a??!?I??/rA@)-C??6:?1??uO??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.