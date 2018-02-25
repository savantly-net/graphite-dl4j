package net.savantly.learning.graphite.convert;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

import org.datavec.api.util.ClassPathResource;
import org.junit.Test;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;

import net.savantly.learning.graphite.convert.GraphiteToCsv;
import net.savantly.learning.graphite.domain.GraphiteMultiSeries;

public class GraphiteMultiSeriesToCsvTest {
	private final static Logger log = LoggerFactory.getLogger(GraphiteMultiSeriesToCsvTest.class);
	
	@Test
	public void testConversion() throws JsonParseException, JsonMappingException, IOException {
		ClassPathResource jsonResource = new ClassPathResource("test.json");
		File jsonFile = jsonResource.getFile();
		Pair<String, GraphiteMultiSeries> series = Pair.of("test", GraphiteMultiSeries.from(jsonFile));
		File csvFile = File.createTempFile("data", ".csv");
		GraphiteToCsv.get(csvFile.getParent()).createFile(csvFile.getName(), series);
		log.info(csvFile.getAbsolutePath());
	}
	
	@Test
	public void testSequentialFiles() throws JsonParseException, JsonMappingException, IOException {
		ClassPathResource jsonResource = new ClassPathResource("test.json");
		File jsonFile = jsonResource.getFile();
		Pair<String, GraphiteMultiSeries> series1 = Pair.of("0", GraphiteMultiSeries.from(jsonFile));
		Pair<String, GraphiteMultiSeries> series2 = Pair.of("1", GraphiteMultiSeries.from(jsonFile));
		
		Path dir = Files.createDirectories(Paths.get("target", "datavec"));
		
		GraphiteToCsv.get(dir.toAbsolutePath().toString()).createFileSequence(Arrays.asList(series1, series2));
		Arrays.stream(dir.toFile().list()).forEach(f -> {
			log.info(f);
		});
		
	}


}
