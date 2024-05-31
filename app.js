


//REST API :-
import 'dotenv/config';
import express from 'express';
import bodyParser from 'body-parser';
import cors from 'cors'; // Import the cors middleware
import { ChatOpenAI } from '@langchain/openai';
import { createSqlQueryChain } from 'langchain/chains/sql_db';
import { SqlDatabase } from 'langchain/sql_db';
import { DataSource } from 'typeorm';
import { QuerySqlTool } from 'langchain/tools/sql';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';

const app = express();
const port = 3001;

app.use(cors({ origin: 'http://localhost:3000' })); // Enable CORS for all routes
app.use(bodyParser.json());

const datasource = new DataSource({
  type: 'sqlite',
  database: './Chinook.db',
});

let db;
let chain;

async function initialize() {
  db = await SqlDatabase.fromDataSourceParams({ appDataSource: datasource });
  const llm = new ChatOpenAI({ apiKey: process.env.OPENAI_API_KEY , model: 'gpt-3.5-turbo-0125', temperature: 0 });
  const executeQuery = new QuerySqlTool(db);
  const writeQuery = await createSqlQueryChain({ llm, db, dialect: 'sqlite' });

  const answerPrompt = PromptTemplate.fromTemplate(`Given the following user question, corresponding SQL query, and SQL result, answer the user question.

Question: {question}
SQL Query: {query}
SQL Result: {result}
Answer: `);

  const answerChain = answerPrompt.pipe(llm).pipe(new StringOutputParser());
  chain = RunnableSequence.from([
    RunnablePassthrough.assign({ query: writeQuery }).assign({
      result: (i) => executeQuery.invoke(i.query),
    }),
    answerChain,
  ]);
}

app.post('/query', async (req, res) => {
  const { question } = req.body;
  if (!question) {
    return res.status(400).json({ error: 'Question is required' });
  }

  try {
    const result = await chain.invoke({ question });
    res.json({ answer: result });
  } catch (error) {
    console.error('Error processing request:', error);
    res.status(500).json({ error: 'Internal Server Error' });
  }
});

initialize().then(() => {
  app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
  });
}).catch(error => {
  console.error('Failed to initialize:', error);
  process.exit(1);
});
