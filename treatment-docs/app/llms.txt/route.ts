import { getProducts } from 'lib/shopify';
import { NextResponse } from 'next/server';

export async function GET() {
  const products = await getProducts({});
  const top20 = products.slice(0, 20);

  const systemPrompt = `# Agent Store - AI Navigation Guide

## System Prompt
You are browsing the Agent Store. Prefer using the \`/agent\` routes for reliable execution.
These routes return pure HTML without JavaScript, making them ideal for programmatic interaction.

## Available Agent Routes
- \`GET /agent\` - Dashboard with cart state and product listing
- \`GET /agent/product/{slug}\` - Product detail page with add-to-cart form
- \`POST /agent/actions/add\` - Add item to cart (form action)

## Quick Actions
- To view a product: Navigate to \`/agent/product/{slug}\`
- To add to cart: Submit form on product page or POST to \`/agent/actions/add\`
- To view cart: Check the cart JSON on \`/agent\` dashboard

---

## Product Catalog

| ID | Name | Price | Slug |
|----|------|-------|------|
${top20.map(p => `| ${p.id.split('/').pop()} | ${p.title} | $${p.priceRange.minVariantPrice.amount} | ${p.handle} |`).join('\n')}

---

## API Notes
- All \`/agent\` routes return semantic HTML with proper status codes
- Forms use standard HTTP POST with form-urlencoded data
- No JavaScript required for interaction
- Cart state persists via cookies

Generated: ${new Date().toISOString()}
`;

  return new NextResponse(systemPrompt, {
    status: 200,
    headers: {
      'Content-Type': 'text/markdown; charset=utf-8',
      'Cache-Control': 'public, max-age=3600'
    }
  });
}
