// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		starlight({
			title: 'BioShift',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/chlofisher/bioshift' }],
			sidebar: [
				{
					label: 'Guides',
					items: [
						// Each item here is one entry in the navigation menu.
						{ label: 'Getting Started', slug: 'guides/getting_started' },
					],
				},
				{
					label: 'Reference',
					autogenerate: { directory: 'reference' },
				},
			],
            editLink: {
                baseUrl: 'https://github.com/chlofisher/bioshift/edit/main/docs'
            },
            customCss: [
                './src/styles/style.css'
            ]
		}),
	],
});
