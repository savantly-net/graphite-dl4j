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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.test.context.junit4.SpringRunner;

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
				.setNumberOfIterations(3)
				.setLearningRate(0.007)
				.build();
		
		MultiLayerNetwork result = classifier.train();
		log.debug(result.summary());
		
		//Save the model
        File locationToSave = new File("target/MyMultiLayerNetwork.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(result, locationToSave, saveUpdater);
	}

}
