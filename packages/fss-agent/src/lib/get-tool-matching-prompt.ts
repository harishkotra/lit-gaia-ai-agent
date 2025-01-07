// Import the FssTool type from the '@lit-protocol/fss-tool' package.
import type { FssTool } from '@lit-protocol/fss-tool';

/**
 * Generates a prompt for OpenAI to analyze a user's intent and match it to an appropriate tool.
 * The prompt includes descriptions of the available tools and instructions for OpenAI to follow.
 *
 * @param tools - An array of FssTool objects representing the available tools.
 * @returns A string containing the generated prompt.
 */
export function getToolMatchingPrompt(tools: FssTool[]): string {
  // Generate a formatted list of tool descriptions, including their name, description, and IPFS CID.
  const toolDescriptions = tools
    .map(
      (tool) =>
        `- Name: ${tool.name}\n  Description: ${tool.description}\n  CID: ${tool.ipfsCid}`
    )
    .join('\n');

  // Return the full prompt, including instructions for OpenAI and the list of available tools.
  return `You are a web3 transaction analyzer. Given a user's intent and available tools, determine if there's an appropriate tool that matches exactly what the user wants to do.
        Available tools:
        ${toolDescriptions}
        Important:
        1. Only return a recommendedCID if you are completely certain the tool matches the user's intent exactly
        2. If you're unsure or the user's intent is unclear, return an empty recommendedCID
        3. All values in your response must be strings
        
        Return a JSON object with:
        {
          "recommendedCID": "the IPFS CID of the recommended tool, or empty string if no confident match",
          "reasoning": "explanation of why this tool was chosen or why no tool was chosen"
        }`;
}
