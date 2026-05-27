// server.js - OpenAI to Modal API Proxy (Optimized for GLM 5.1-FP8 & Janitor AI)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Allows massive contexts without crashing

// MODAL API Configuration
const MODAL_API_BASE = process.env.MODAL_API_BASE || 'https://api.us-west-2.modal.direct/v1';
const MODAL_API_KEY = process.env.MODAL_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Set to true to show <think> blocks in Janitor AI
const SHOW_REASONING = true; 

// 🛑 KEEP FALSE - Modal's standard engine will reject custom Nvidia arguments
const ENABLE_THINKING_MODE = false;

// Model mapping: Type any of these left names into Janitor AI, and it forces GLM 5.1
const MODEL_MAPPING = {
  'gpt-4': 'zai-org/GLM-5.1-FP8',
  'gpt-4o': 'zai-org/GLM-5.1-FP8',
  'gpt-3.5-turbo': 'zai-org/GLM-5.1-FP8',
  'claude-3-opus': 'zai-org/GLM-5.1-FP8',
  'glm-5': 'zai-org/GLM-5.1-FP8',
  'zai-org/GLM-5.1': 'zai-org/GLM-5.1-FP8'
};

// Trim history safely to avoid memory overloads up to ~32k tokens
function truncateMessages(messages, maxChars = 130000) {
  const cleanedMessages = messages.map(msg => {
    if (msg.role === 'assistant' && msg.content) {
      return { ...msg, content: msg.content.replace(/<think>[\s\S]*?<\/think>\n*/g, '').trim() };
    }
    return msg;
  });

  if (JSON.stringify(cleanedMessages).length <= maxChars) return cleanedMessages;

  const systemMsgs = cleanedMessages.filter(m => m.role === 'system');
  let convoMsgs = cleanedMessages.filter(m => m.role !== 'system');

  while (JSON.stringify([...systemMsgs, convoMsgs]).length > maxChars && convoMsgs.length > 2) {
    convoMsgs.shift();
  }
  return [...systemMsgs, convoMsgs];
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to Modal GLM Proxy', 
    reasoning_display: SHOW_REASONING
  });
});

// List models endpoint
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'modal-proxy'
  }));
  res.json({ object: 'list', data: models });
});

// Main Proxy Chat Completions
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Default to the target Modal model if the map misses
    let selectedModel = MODEL_MAPPING[model] || 'zai-org/GLM-5.1-FP8';
    const trimmedMessages = truncateMessages(messages);
    
    const modalRequest = {
      model: selectedModel,
      messages: trimmedMessages,
      temperature: temperature || 0.8,
      max_tokens: max_tokens || 4096,
      stream: stream || false
    };
    
    const response = await axios.post(`${MODAL_API_BASE}/chat/completions`, modalRequest, {
      headers: {
        'Authorization': `Bearer ${MODAL_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        lines.forEach(line => {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write(line + '\n');
              return;
            }
            
            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const reasoning = data.choices[0].delta.reasoning_content;
                const content = data.choices[0].delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  if (reasoning && !reasoningStarted) {
                    combinedContent = '<think>\n' + reasoning;
                    reasoningStarted = true;
                  } else if (reasoning) {
                    combinedContent = reasoning;
                  }
                  
                  if (content && reasoningStarted) {
                    combinedContent += '</think>\n\n' + content;
                    reasoningStarted = false;
                  } else if (content) {
                    combinedContent += content;
                  }
                  
                  if (combinedContent) {
                    data.choices[0].delta.content = combinedContent;
                    delete data.choices[0].delta.reasoning_content;
                  }
                } else {
                  if (content) data.choices[0].delta.content = content;
                  delete data.choices[0].delta.reasoning_content;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        });
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());
    } else {
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          return {
            index: choice.index,
            message: { role: choice.message.role, content: fullContent },
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: { message: error.message || 'Internal server error', code: error.response?.status || 500 }
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({ error: { message: `Endpoint ${req.path} not found`, code: 404 } });
});

app.listen(PORT, () => {
  console.log(`OpenAI to Modal GLM Proxy live on port ${PORT}`);
});
