?!  *	????̼o@2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2Q?|a2??!?h?=N@@)?|гY???1?Ѡ??3@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map$(~????!?}???SE@)J+???1G:??L3@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice 8gDio??!h?^?I)@)8gDio??1h?^?I)@:Preprocessing2?
]Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice j?t???!N?ׯ?? @)j?t???1N?ׯ?? @:Preprocessing2?
PIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::Zip[1]::BatchV2f??a?֤?!?8'??0@)a2U0*???1s??d??@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV22??%䃞?!U???Ky'@)???Q???1???6??@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[0]::BatchV2::TensorSlice ???_vO??!??!??P@)???_vO??1??!??P@:Preprocessing2?
UIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2=?U?????!{k???"@) ?o_Ή?1,?;???@:Preprocessing2?
bIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip[1]::BatchV2::TensorSlice Zd;?O???!????@)Zd;?O???1????@:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTake::ZipF%u???!n?c#??I@)a??+ey?1`?&?@:Preprocessing2b
+Iterator::Model::MaxIntraOpParallelism::ZipZd;?O???!??fIxEX@)"??u??q?1~x????:Preprocessing2F
Iterator::Model??????!      Y@)????Mbp?1??p)?4??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismx??#????!?=ZC,?X@)_?Q?k?1(??|?l??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake::Zip?46<??!H_-:}6@)-C??6j?1????X*??:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::Zip[0]::Mapk?w??#??!Ej&?_^J@)?~j?t?h?1?w_????:Preprocessing2v
?Iterator::Model::MaxIntraOpParallelism::Zip[0]::Map::FiniteTakeo???T???!??-."?I@)a2U0*?S?1s??d????:Preprocessing2o
8Iterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map?;Nё\??!????[7@)a2U0*?S?1s??d????:Preprocessing2{
DIterator::Model::MaxIntraOpParallelism::Zip[1]::Map::Map::FiniteTake:??H???!?
??6@)????MbP?1??p)?4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q??M?{=U@"?
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
Refer to the TF2 Profiler FAQb?85.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.JDESKTOP-AR1HCVR: Failed to load libcupti (is it installed and accessible?)