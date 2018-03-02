package net.savantly.learning.graphite.learners;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

public interface MultiLayerLearner {
	
	DataSetIterator getTestingDataSets();
	DataSetIterator getTrainingDataSets();
	MultiLayerNetwork train();
	MultiLayerConfiguration createNetworkConfiguration();
}
