package net.savantly.learning.graphite;

import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.test.context.junit4.SpringRunner;

import net.savantly.graphite.GraphiteClientFactory;
import net.savantly.graphite.query.GraphiteQueryBuilder;

@RunWith(SpringRunner.class)
public class GraphiteBooleanClassificationTest {
	private static Logger log = LoggerFactory.getLogger(GraphiteBooleanClassificationTest.class);
	
	@Value("#{systemEnvironment['GRAPHITE_HOST']}")
	String graphiteHost;
	
	@Test
	public void test() throws UnknownHostException, SocketException, IOException {
		GraphiteBooleanClassification classifier = GraphiteBooleanClassification.builder()
				.setWorkingDirectory(Files.createDirectories(Paths.get("target/data")).toFile())
				.addNegativeQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(-1), 'A')"))
				.addNegativeQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(-1), 'B')"))
				.addPositiveQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(1), 'A')"))
				.addPositiveQuery(GraphiteQueryBuilder.simpleQuery("alias(constantLine(1), 'B')"))
				.setNumberOfIterations(5)
				.setClient(GraphiteClientFactory.queryableGraphiteClient(graphiteHost))
				.build();
		MultiLayerNetwork network = classifier.train();
		log.info(network.summary());
		
		INDArray prediction = classifier.evaluate(GraphiteQueryBuilder.simpleQuery("alias(constantLine(1), 'Z')"));
		log.info(prediction.toString());
	}

}
