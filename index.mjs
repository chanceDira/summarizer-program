import * as dotenv from "dotenv";
import { OpenAI } from "langchain";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import * as fs from "fs";

dotenv.config();

const run = async () => {
  console.log(process.cwd())
  const text = fs.readFileSync("pilot.txt", "utf8");
  const model = new OpenAI({ temperature: 0 });
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
  const docs = await textSplitter.createDocuments([text]);
  const chain = loadSummarizationChain(model, { type: "map_reduce" });
  const res = await chain.call({
    input_documents: docs,
  });
  console.log({ res });
 
};

run()