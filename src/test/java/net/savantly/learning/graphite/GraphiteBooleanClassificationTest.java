package net.savantly.learning.graphite;

import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.test.context.junit4.SpringRunner;

@RunWith(SpringRunner.class)
public class GraphiteBooleanClassificationTest {
	private static Logger log = LoggerFactory.getLogger(GraphiteBooleanClassificationTest.class);
	
	@Value("#{systemEnvironment['GRAPHITE_HOST']}")
	String graphiteHost;
	
	@Test
	public void test() throws UnknownHostException, SocketException, IOException {
		/*GraphiteBooleanClassification classifier = GraphiteBooleanClassification.builder()
				.setWorkingDirectory(Files.createDirectories(Paths.get("target/data")).toFile())
				.addNegativeQuery(GraphiteQueryBuilder.csvQuery("alias(constantLine(-1), 'A')"))
				.addNegativeQuery(GraphiteQueryBuilder.csvQuery("alias(constantLine(-1), 'A')"))
				.addPositiveQuery(GraphiteQueryBuilder.csvQuery("alias(constantLine(1), 'B')"))
				.addPositiveQuery(GraphiteQueryBuilder.csvQuery("alias(constantLine(1), 'B')"))
				.setNumberOfIterations(5)
				.setClient(GraphiteClientFactory.queryableGraphiteClient(graphiteHost))
				.build();
		MultiLayerNetwork network = classifier.train();
		log.info(network.summary());
		
		INDArray shouldBePositivePrediction = classifier.evaluate(GraphiteQueryBuilder.simpleQuery("alias(constantLine(1), 'Y')"));
		INDArray shouldBeNegativePrediction = classifier.evaluate(GraphiteQueryBuilder.simpleQuery("alias(constantLine(-1), 'Z')"));
		
		log.info("positive: {}, negative: {}", shouldBePositivePrediction, shouldBeNegativePrediction);*/
		// TODO: Implement this
	}

}
