package net.savantly.learning.graphite.domain;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import org.nd4j.linalg.dataset.api.DataSet;

import com.fasterxml.jackson.core.JsonParseException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

public class GraphiteMultiSeries extends ArrayList<GraphiteSeries> {
	private static final ObjectMapper mapper = new ObjectMapper();
	
	public static GraphiteMultiSeries from(File file) throws JsonParseException, JsonMappingException, IOException {
		StringBuilder sb = new StringBuilder();
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.lines().forEach(s -> {
			sb.append(s);
		});
		reader.close();
		return from(sb.toString());
	}
	
	public static GraphiteMultiSeries from(JsonNode jsonNode) {
		return mapper.convertValue(jsonNode, GraphiteMultiSeries.class);
	}

	public static GraphiteMultiSeries from(String json) throws JsonParseException, JsonMappingException, IOException {
		return mapper.readValue(json, GraphiteMultiSeries.class);
	}

	public static GraphiteMultiSeries from(InputStream json) throws JsonParseException, JsonMappingException, IOException {
		return mapper.readValue(json, GraphiteMultiSeries.class);
	}
	
}
