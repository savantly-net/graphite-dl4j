package net.savantly.learning.graphite.learners.timeseries;

import java.io.File;
import java.io.IOException;
import java.util.List;

import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.test.context.junit4.SpringRunner;

import net.savantly.learning.graphite.domain.GraphiteCsv;

@RunWith(SpringRunner.class)
public class GraphitePredictionTest {

private static final Logger log = LoggerFactory.getLogger(GraphitePredictionTest.class);
	
	@Value("classpath:/data/training/graphitePrediction/randomWalk01.csv")
	Resource exampleFile;
	@Value("classpath:/data/training/graphitePrediction/randomWalk02.csv")
	Resource exampleFile2;

	@Test
	public void test() throws IOException {
		final int lagSize = 3;
		List<List<List<Writable>>> trainingData = GraphiteCsv.from(exampleFile.getFile()).asRecords(lagSize);
		GraphitePredictor predictor = GraphitePredictor.builder()
				.setTrainingData(trainingData)
				.setEpochs(10)
				.setLearningRate(0.05)
				.setNumOfInputs(lagSize)
				.build();
		MultiLayerNetwork network = predictor.train();
		log.info(network.summary());
		
		// example for prediction based on new data
		boolean includeLabels = false;
		INDArray testingData = GraphiteCsv.from(exampleFile2.getFile()).as3dMatrix(lagSize, includeLabels);
		// How many timesteps do we want to predict past the end of our data
		int timeSteps = 30;
		
		float[] predictedValues = predictor.doTimeStep(timeSteps, testingData);
		for (int i = 0; i < predictedValues.length; i++) {
			String formatted = String.format("timestep: %s: %s\n", i, predictedValues[i]);
			log.info(formatted);
		}
		//log.info(predictor.predict(timeSteps).toString());
		
	}
	
	@Test
	public void testSavedNetwork() throws IOException {
		final int lagSize = 2;
		List<List<List<Writable>>> trainingData = GraphiteCsv.from(exampleFile.getFile()).asRecords(lagSize);
		GraphitePredictor predictor = GraphitePredictor.builder()
				.setTrainingData(trainingData)
				.setEpochs(10)
				.setLearningRate(0.05)
				.setNumOfInputs(lagSize)
				.build();
		MultiLayerNetwork network = predictor.train();
		log.info(network.summary());
		
		boolean saveUpdater = true;
		File file = File.createTempFile("trainedModel", ".net");
		// save trained network
		ModelSerializer.writeModel(network, file, saveUpdater);
		// load saved network
		MultiLayerNetwork restoredNetwork = ModelSerializer.restoreMultiLayerNetwork(file, saveUpdater);
		// rebuild predictor with saved network
		predictor = predictor.build(restoredNetwork);
		
		// example for prediction based on new data
		boolean includeLabels = false;
		INDArray testingData = GraphiteCsv.from(exampleFile2.getFile()).as3dMatrix(lagSize, includeLabels);
		// How many timesteps do we want to predict past the end of our data
		int timeSteps = 30;
		
		float[] predictedValues = predictor.doTimeStep(timeSteps, testingData);
		for (int i = 0; i < predictedValues.length; i++) {
			String formatted = String.format("timestep: %s: %s\n", i, predictedValues[i]);
			log.info(formatted);
		}
		//log.info(predictor.predict(timeSteps).toString());
		
	}
}
