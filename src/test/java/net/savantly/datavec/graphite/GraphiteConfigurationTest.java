package net.savantly.datavec.graphite;

import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator.AlignmentMode;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.test.context.junit4.SpringRunner;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.sprint.eai.web.configuration.GraphiteConfiguration;

import net.savantly.datavec.graphite.GraphiteMultiSeries;
import net.savantly.datavec.graphite.GraphiteMultiSeriesToCsv;
import net.savantly.graphite.QueryableGraphiteClient;
import net.savantly.graphite.query.GraphiteQuery;
import net.savantly.graphite.query.GraphiteQueryBuilder;
import net.savantly.graphite.query.fomat.JsonFormatter;

@RunWith(SpringRunner.class)
public class GraphiteConfigurationTest {
	private static final Logger log = LoggerFactory.getLogger(GraphiteConfigurationTest.class);

	@Autowired
	GraphiteConfiguration graphiteConfig;

	@Value("classpath:/data/training/bad_*.json")
	Resource[] baddies;

	@Value("classpath:/data/training/good_*.json")
	Resource[] goodies;

	@Test
	public void testClient() throws UnknownHostException, SocketException {
		QueryableGraphiteClient client = graphiteConfig.graphiteClient();
		JsonFormatter formatter = new JsonFormatter();
		GraphiteQuery<JsonNode> query = new GraphiteQueryBuilder<>(formatter).setTarget("constantLine(123.456)")
				.build();
		JsonNode results = client.query(query);
		log.debug("{}", results);
	}

	@Test
	public void testBTData() throws UnknownHostException, SocketException {
		QueryableGraphiteClient client = graphiteConfig.graphiteClient();
		JsonFormatter formatter = new JsonFormatter();
		GraphiteQuery<JsonNode> query = new GraphiteQueryBuilder<>(formatter).setTarget("constantLine(123.456)")
				.build();
		JsonNode results = client.query(query);
		log.debug("{}", results);
	}

	@Test
	public void testSequentialFiles()
			throws JsonParseException, JsonMappingException, IOException, InterruptedException {

		Path dir = Files.createDirectories(Paths.get("target", "datavec"));

		List<Pair<String, GraphiteMultiSeries>> pairs = new ArrayList<>();

		// Add the 'good' examples and label them 0, as in the outage condition
		// was 'false'
		Arrays.stream(this.goodies).forEach(g -> {
			try {
				Pair p = Pair.of("0", GraphiteMultiSeries.from(g.getFile()));
				pairs.add(p);
			} catch (IOException e) {
				log.error("{}", e);
			}
		});

		// Add the 'bad' examples and label them 1, as in the outage condition was
		// 'true'
		Arrays.stream(this.baddies).forEach(g -> {
			try {
				Pair p = Pair.of("1", GraphiteMultiSeries.from(g.getFile()));
				pairs.add(p);
			} catch (IOException e) {
				log.error("{}", e);
			}
		});

		// Write the transformed data to csv files in the dir
		int fileCount = GraphiteMultiSeriesToCsv.get(dir.toAbsolutePath().toString()).createFileSequence(pairs);

		Arrays.stream(dir.toFile().list()).forEach(f -> {
			log.info(f);
		});
		

		String featureFilesMatcher = dir.resolve("%d.features.csv").toAbsolutePath().toString();
		String labelFilesMatcher = dir.resolve("%d.labels.csv").toAbsolutePath().toString();
		
		// ----- Load the training data -----
		SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(0,",");
		
		trainFeatures.initialize(new NumberedFileInputSplit(featureFilesMatcher , 0, fileCount - 2));
		SequenceRecordReader trainLabels = new CSVSequenceRecordReader(0,",");
		trainLabels.initialize(new NumberedFileInputSplit(labelFilesMatcher, 0, fileCount - 2));

		int miniBatchSize = 1;
		int numberOfPossibleLabels = 2;
		boolean doRegression = false;
		
		DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize,
				numberOfPossibleLabels, doRegression, AlignmentMode.ALIGN_END);
		
		

		// Normalize the training data
		DataNormalization normalizer = new NormalizerStandardize();
		normalizer.fit(trainData); // Collect training data statistics
		trainData.reset();

		// Use previously collected statistics to normalize on-the-fly. Each DataSet
		// returned by 'trainData' iterator will be normalized
		trainData.setPreProcessor(normalizer);

		// ----- Load the test data -----
		// Same process as for the training data.
		SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
		testFeatures
				.initialize(new NumberedFileInputSplit(featureFilesMatcher, 0, fileCount - 1));
		SequenceRecordReader testLabels = new CSVSequenceRecordReader();
		testLabels.initialize(new NumberedFileInputSplit(labelFilesMatcher, 0, fileCount - 1));

		DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize,
				numberOfPossibleLabels, false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

		//testData.setPreProcessor(normalizer); // Note that we are using the exact same normalization process as the
												// training data
		

		// ----- Configure the network -----
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(123) // Random number generator seed
																						// for improved repeatability.
																						// Optional.
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS).learningRate(0.005)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) // Not always required, but
																							// helps with this data set
				.gradientNormalizationThreshold(0.5).list()
				.layer(0, new GravesLSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build())
				.layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation(Activation.SOFTMAX)
						.nIn(10).nOut(numberOfPossibleLabels).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();

		net.setListeners(new ScoreIterationListener(20)); // Print the score (loss function value) every 20 iterations

		// ----- Train the network, evaluating the test set performance at each epoch
		// -----
		int nEpochs = 40;
		String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
		for (int i = 0; i < nEpochs; i++) {
			net.fit(trainData);

			// Evaluate on the test set:
			Evaluation evaluation = net.evaluate(testData);
			log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));

			testData.reset();
			trainData.reset();
		}

		log.info("----- Example Complete -----");

	}

	@Configuration
	static class TestConfig {

		@Bean
		public GraphiteConfiguration graphiteConfiguration() {
			GraphiteConfiguration config = new GraphiteConfiguration();
			config.setHost("144.229.218.107");
			return config;
		}
	}

}
