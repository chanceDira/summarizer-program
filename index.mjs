// import * as dotenv from "dotenv";
// import { OpenAI } from "langchain";
// import { loadSummarizationChain } from "langchain/chains";
// import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
// import * as fs from "fs";

// dotenv.config();

// const run = async () => {
//   console.log(process.cwd())
//   const text = fs.readFileSync("pilot.txt", "utf8");
//   const model = new OpenAI({ temperature: 0 });
//   const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
//   const docs = await textSplitter.createDocuments([text]);
//   const chain = loadSummarizationChain(model, { type: "map_reduce" });
//   const res = await chain.call({
//     input_documents: docs,
//   });
//   console.log({ res });
 
// };

// run()

import { OpenAI } from "langchain/llms/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { HNSWLib } from "langchain/vectorstores/hnswlib";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as dotenv from "dotenv";
import * as fs from "fs";
dotenv.config();

export const run = async () => {
  /* Initialize the LLM to use to answer the question */
  const model = new OpenAI({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName:"gpt-3.5-turbo"
  });
  /* Load in the file we want to do question answering over */
  const text = fs.readFileSync("pilot.txt", "utf8");
  /* Split the text into chunks */
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  /* Create the vectorstore */
  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
    modelName:"text-embedding-ada-002"
  }));
  /* Create the chain */
  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorStore.asRetriever()
  );
  /* Ask it a question */
  const question = " Your task is to summarize the TV script I give you and extract name of all Actors and use bulletpoint to list them. pick good emoji to represent each actor.";
  const res = await chain.call({ question, chat_history: [] });
  console.log(res);
  /* Ask it a follow up question */
  // const chatHistory = question + res.text;
  
};

run()
