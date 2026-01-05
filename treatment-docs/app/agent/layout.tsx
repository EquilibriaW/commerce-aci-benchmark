import { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Agent Interface - Agent Store',
  description: 'AI-friendly interface for Agent Store',
  robots: 'noindex, nofollow'
};

export default function AgentLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Return children directly - the pages will handle their own HTML structure
  return children;
}
