  *	??????g@2j
3Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map?C??????!$ظѰGJ@)??JY?8??1??ά6@:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV25?8EGr??!O?\?p?9@)%u???1iCT???.@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice Dio??ɔ?!6?d?M6%@)Dio??ɔ?16?d?M6%@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2h??|?5??!?s?S?.@)X9??v???1?;?]J2 @:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2-C??6??!?F??*@)?!??u???1?ےw@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice y?&1???!????B@)y?&1???1????B@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2=?U?????!????.)@)???<,Ԋ?1??{??`@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice Zd;?O???!?q?a?@)Zd;?O???1?q?a?@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice ?I+???!??o??@)?I+???1??o??@:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip'1?Z??!I?J??D@)?J?4q?1?&0<?@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map
h"lxz??!T???E@)?~j?t?h?1t???:Preprocessing2F
Iterator::Model?):????!      Y@)a2U0*?c?1*u???:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::ZipΪ??V???!?????iX@)a2U0*?c?1*u???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??x?&1??!,??̿?X@)?J?4a?1?&0<???:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip?"??~j??!X?؎2?<@)ŏ1w-!_?1.??T???:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake??ܵ??!?ƍ?="E@)Ǻ???V?1???h??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map???QI??!3)^ ??=@)/n??R?1?@&?d??:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake?w??#???!+?4?rO=@)a2U0*?C?1*u???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.