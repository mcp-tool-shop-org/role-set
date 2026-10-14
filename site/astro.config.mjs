// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import tailwindcss from '@tailwindcss/vite';

// https://astro.build/config
export default defineConfig({
  site: 'https://mcp-tool-shop-org.github.io',
  base: '/role-set',
  integrations: [
    starlight({
      title: 'role-set',
      description: 'Prototype. Plans role-shaping items, writes the parts a writer should write, records the gates each item passed, and builds kind B puzzles in code that are proven before a model words them.',
      logo: {
        src: './src/assets/logo.png',
        alt: 'role-set',
        href: '/role-set/',
        replacesTitle: false,
      },
      disable404Route: true,
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/mcp-tool-shop-org/role-set' },
      ],
      sidebar: [
        {
          label: 'Handbook',
          items: [{ autogenerate: { directory: 'handbook' } }],
        },
      ],
      customCss: ['./src/styles/starlight-custom.css'],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
