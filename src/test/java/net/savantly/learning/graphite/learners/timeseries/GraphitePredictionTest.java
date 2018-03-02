package net.savantly.learning.graphite.learners.timeseries;

import java.io.IOException;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
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
	
	@Value("classpath:/data/training/graphitePrediction/average_response_time.csv")
	Resource exampleFile;

	@Test
	public void test() throws IOException {
		List<INDArray> trainingData = GraphiteCsv.from(exampleFile.getFile()).asINDArray();
		GraphitePredictor predictor = GraphitePredictor.builder()
				.addTrainingData(trainingData)
				.setMiniBatchSize(1000)
				.setEpochs(50)
				
				.build();
		MultiLayerNetwork network = predictor.train();
		log.info(network.summary());
		
		// How many timesteps do we want to predict past the end of our data
		int timeSteps = 2;
		log.info(predictor.doTimeStep(timeSteps).toString());
		log.info(predictor.predict(timeSteps).toString());
		
	}
}
