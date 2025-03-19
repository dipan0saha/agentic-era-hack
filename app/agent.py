# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import pandas as pd
import streamlit as st
from google.cloud import bigquery

# mypy: disable-error-code="union-attr"
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

logging.basicConfig(level=logging.DEBUG)

LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"


# 1. Define functions
def run_bigquery_query(query):
    """Runs DML and DDL BigQuery query and returns the results"""
    client = bigquery.Client()
    if (
        "drop" in query.lower()
        or "delete" in query.lower()
        or "truncate" in query.lower()
    ):
        return f"Deletion actions are not permitted."
    else:
        try:
            query_job = client.query(query)
            results = query_job.result()
            # Convert the results to a DataFrame directly
            return results.to_dataframe()
        except Exception as e:
            return f"Error running BigQuery query: {e}"


# 2. Define tools
@tool
def bigquery_tool(query: str):
    """Tool to run DML and DDL BigQuery queries."""
    return run_bigquery_query(query)


tools = [bigquery_tool]

# 3. Set up the language model
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0, max_tokens=1024, streaming=True
).bind_tools(tools)


# 4. Define workflow components
def should_continue(state: MessagesState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Calls the language model and returns the response."""
    system_message = "You are a helpful AI assistant."
    messages_with_system = [{"type": "system", "content": system_message}] + state[
        "messages"
    ]
    # Forward the RunnableConfig object to ensure the agent is capable of streaming the response.
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


# 5. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# 6. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 7. Compile the workflow
agent = workflow.compile()
