package net.savantly.learning.graphite.learners.timeseries;

import java.net.SocketException;
import java.net.UnknownHostException;

import net.savantly.graphite.GraphiteClientFactory;
import net.savantly.graphite.QueryableGraphiteClient;

public class PersistentGraphitePredictorTest {
	
	
	private static final String graphiteHost = "127.0.0.1";
	private static final int windowSize = 2;
	private static final String target = "randomWalk()";
	private static final String targetAlias = "random.walk";
	private static final int epochs = 20;
	private static final int hiddenLayerWidth = 30;
	private static final double learningRate = 0.001;

	public void testTraining() throws UnknownHostException, SocketException {
		QueryableGraphiteClient client = GraphiteClientFactory.queryableGraphiteClient(graphiteHost);
		PersistentGraphitePredictor learner = 
				PersistentGraphitePredictor.builder(client )
				.setWindowSize(windowSize)
				.setTrainingTarget(target)
				.setTestingTarget(target)
				.setTargetAlias(targetAlias)
				.setEpochs(epochs)
				.setHiddenLayerWidth(hiddenLayerWidth)
				.setLearningRate(learningRate)
				.setRnnLearningRate(learningRate)
				.setUseCache(false) // you can cache/store the model to re-use between app restarts
				.build();
		
		learner.train(); // train the network using the graphite targets
		learner.update(); // make a 2 min prediction and store it in graphite
	}

}
