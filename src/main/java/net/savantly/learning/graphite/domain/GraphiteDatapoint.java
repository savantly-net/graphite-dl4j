package net.savantly.learning.graphite.domain;

import java.util.ArrayList;

public class GraphiteDatapoint extends ArrayList<Number> implements Comparable<GraphiteDatapoint> {
	
	private String label;
	public String getLabel() {
		return label;
	}
	public void setLabel(String label) {
		this.label = label;
	}
	
	public GraphiteDatapoint() {
		super(2);
	}

	public Number getValue() {
		return this.get(0);
	}
	public void setValue(Number value) {
		this.set(0, value);
	}
	
	public Number getEpoc() {
		return this.get(1);
	}
	public void setEpoc(Number value) {
		this.set(1, value);
	}

	/**
	 * naturally ordered by epoch
	 * @param o
	 * @return
	 */
	@Override
	public int compareTo(GraphiteDatapoint o) {
		return Long.compare(this.getEpoc().longValue(), o.getEpoc().longValue());
	}
}
