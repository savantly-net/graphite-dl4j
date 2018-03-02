package net.savantly.learning.graphite.learners.timeseries;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.test.context.junit4.SpringRunner;

import net.savantly.learning.graphite.convert.GraphiteToDataSet;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;

@RunWith(SpringRunner.class)
public class GraphiteSeriesClassifierTest {
	
	private static final Logger log = LoggerFactory.getLogger(GraphiteSeriesClassifierTest.class);
	
	@Value("classpath:/data/training/graphite/bad_*.json")
	Resource[] baddies;

	@Value("classpath:/data/training/graphite/good_*.json")
	Resource[] goodies;

	@Test
	public void test() throws IOException, InterruptedException {
		File dir = Files.createDirectories(Paths.get("target", "datavec")).toFile();

		List<GraphiteMultiSeries> positiveTrainingExamples = new ArrayList<>();
		List<GraphiteMultiSeries> negativeTrainingExamples = new ArrayList<>();

		// Add the 'negative condition' examples
		Arrays.stream(this.goodies).forEach(g -> {
			try {
				negativeTrainingExamples.add(GraphiteMultiSeries.from(g.getFile()));
			} catch (IOException e) {
				log.error("{}", e);
			}
		});

		// Add the 'positive condition' examples
		Arrays.stream(this.baddies).forEach(g -> {
			try {
				positiveTrainingExamples.add(GraphiteMultiSeries.from(g.getFile()));
			} catch (IOException e) {
				log.error("{}", e);
			}
		});
		
		GraphiteSeriesClassifier classifier = GraphiteSeriesClassifier.builder()
				.setWorkingDirectory(dir)
				.setPositiveExamples(positiveTrainingExamples)
				.setNegativeExamples(negativeTrainingExamples)
				.setNumberOfIterations(5)
				.setLearningRate(0.007)
				.setDoRegression(true)
				.build();
		
		MultiLayerNetwork net = classifier.train();
		log.debug(net.summary());
		
		boolean shouldBePositive = testSample(classifier, net, positiveTrainingExamples);
		boolean shouldBeNegative = testSample(classifier, net, negativeTrainingExamples);

		
		//Save the model
        File locationToSave = new File("target/MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater);
	}

	private boolean testSample(GraphiteSeriesClassifier classifier, MultiLayerNetwork net, List<GraphiteMultiSeries> exampleList) {
		Pair[] pairs = new Pair[]{ Pair.of(0, exampleList.get(0))};
		
		List<Pair<INDArray, INDArray>> testPairs = GraphiteToDataSet
				.toTimeSeriesNDArrayPairs(pairs);
		DataSetIterator posIter = classifier.createDataSetIterator(testPairs);
	    INDArray timeSeriesOutput = net.output(posIter);
	    int timeSeriesLength = timeSeriesOutput.size(2);
	    
	    INDArray lastTimeStepProbabilities = timeSeriesOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(timeSeriesLength-1));
	    log.debug(lastTimeStepProbabilities.toString());
	    
	    INDArray rnnTimeStep = net.rnnTimeStep(testPairs.get(0).getLeft());
	    
		return false;
	}

}
