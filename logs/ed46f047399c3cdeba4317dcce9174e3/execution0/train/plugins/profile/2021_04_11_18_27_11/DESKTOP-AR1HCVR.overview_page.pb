?!  *	?????,k@2j
3Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map|a2U0*??!y ?(??F@)46<?R??1??Gq!4@:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2?S㥛İ?!8???? >@)M?J???1?-BY;?1@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice F%u???!?u?zI(@)F%u???1?u?zI(@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice Dio??ɔ?!Daݬ"@)Dio??ɔ?1Daݬ"@:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2Y?8??m??!w?ˡoZ2@)n????1?^S?"@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2S?!?uq??!?????(@)_?Qڋ?1???D?@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice F%u???!?u?zI@)F%u???1?u?zI@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2??0?*??!?????%@)?~j?t???1lJ< A@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice ???????!?6)o?W@)???????1?6)o?W@:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip?Q???!?????@I@)/n??r?1
??w?0 @:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip$???~???!?Sd*?=8@)HP?s?b?1???e???:Preprocessing2F
Iterator::ModelO@a????!      Y@)/n??b?1
??w?0??:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::Zip??{??P??!?T?>?X@)/n??b?1
??w?0??:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map鷯???!?	i>I?I@)????Mb`?1???Vp??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??q????!Hq!<?X@)??H?}]?1???f?~??:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Mapı.n???!F,??6)9@)Ǻ???V?1T#?o???:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake?[ A?c??!p?㇁I@)/n??R?1
??w?0??:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake???9#J??!,{?`[?8@)a2U0*?C?1#?əͩ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q/?T?:N@"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?60.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-AR1HCVR: Failed to load libcupti (is it installed and accessible?)