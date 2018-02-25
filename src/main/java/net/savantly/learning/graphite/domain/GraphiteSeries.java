package net.savantly.learning.graphite.domain;

import java.util.ArrayList;
import java.util.List;

public class GraphiteSeries {

	private String target;
	private List<GraphiteDatapoint> datapoints = new ArrayList<>();
	
	public String getTarget() {
		return target;
	}
	public void setTarget(String target) {
		this.target = target;
	}
	public List<GraphiteDatapoint> getDatapoints() {
		return datapoints;
	}
	public void setDatapoints(List<GraphiteDatapoint> datapoints) {
		this.datapoints = datapoints;
	}
}
