# Design brief — Satellite Super Resolution (ORBIT)

## Confirmed direction
Treat the user's supplied landing-page screenshot as the ground-truth visual reference: a bright, clean top navigation; a dark edge-to-edge satellite-terrain hero; bold left-aligned “Satellite Super Resolution” headline; short explanatory copy; four compact application chips; a high-contrast green “Get Started” CTA; and a floating, perspective-treated zoom panel over the map. Follow it with a white “What You Can Do” section with four equally weighted workflow cards. Keep the existing ORBIT mark and the underlying research scope.

## Design dimensions
- **Design movement:** Modern Earth-observation product landing page with scientific restraint, tactile map imagery, and subtle 3D depth.
- **Core principles:** Match the reference's light-header/dark-map/light-workflow rhythm; make the four use areas and four-step sequence immediately scannable; keep CTAs clear; distinguish a visual concept from a real model output.
- **Color philosophy:** Paper white and cool gray for the navigation and lower workflow; deep forest / blue-black over the map; fresh jade / emerald CTA and chips; understated slate text and thin glassy borders.
- **Layout paradigm:** One responsive page. Header links Home, About, Sample Images, Help. Wide map-led hero with copy and category chips on the left and a framed zoom inset on the right. Four action cards in a bright section immediately below. Keep the existing explanatory research content and source-attributed sample further down.
- **Signature elements:** A generated, fictional Earth-observation landscape with a river, forest, farmland, and distant urban texture; a 3D-perspective zoom inset that samples the same hero scene; a small, clearly marked target region and connector; Forest Monitoring, Urban Analysis, Agricultural Observation, and Water Body Monitoring chips; four cards labelled Upload Satellite Image, Select Region Type, Generate Super Resolution, and Download Output.
- **Interaction philosophy:** Header items are real in-page links. “Get Started” scrolls to the workflow. Keep the existing keyboard-operable comparison slider. The map layer may respond gently to pointer position, but does not capture gestures; card copy describes stages, not connected model functions. Do not imply upload, inference, or download works unless a backend actually implements it.
- **Animation:** Gentle, low-amplitude hero depth only; disable it for `prefers-reduced-motion`, touch-only, or fine-pointer absence. No forced scrolling.
- **Typography system:** Retain the existing readable sans/monospace system with high-contrast heading hierarchy; prefer short technical annotations over dense UI text.
- **Brand essence:** Finer views, handled responsibly.
- **Brand voice:** Approachable, useful, and scientifically careful.
- **Wordmark/logo:** Preserve the existing ORBIT mark; use it in the header and keep existing favicon/project-icon identity.
- **Signature brand color:** Emerald / jade green, supported by midnight forest and bright neutral surfaces.

## Planned visual assets
1. **Project logo:** Preserve `/assets/orbit-mark.png` and its `app.config.ts` HTTPS logo metadata.
2. **Hero landscape:** One newly generated 16:9, fictional top-down satellite-style scene, with the main terrain on the right and a lower-detail dark-green area behind the left-side copy. No text, labels, borders, or interface elements inside the generated asset. Mark it as illustrative in the page UI.
3. **Sample imagery:** Preserve the credited Copernicus Sentinel-2 Banat Plain image for the existing comparison; keep the before/after wording explicit that it is only a display treatment, not model inference.

## Factual guardrails
Use the supplied project statement: medium-resolution satellite imagery (typically 10–30 m), a Sentinel-2 10 m input example, and a target product below 4 m. That target is an intended objective, not achieved performance. Do not invent a trained model, accuracy metric, result, live satellite feed, or functioning AI inference. Model-inferred details require validation against high-resolution reference data and preservation of geospatial/spectral consistency. Keep “NTRO — confirm affiliation before publishing” as an editable unconfirmed affiliation placeholder and do not use its logo.
