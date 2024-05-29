import{_ as a,c as s,o as i,a6 as t}from"./chunks/framework.C4m5toHo.js";const b=JSON.parse('{"title":"Internals","description":"","frontmatter":{},"headers":[],"relativePath":"private.md","filePath":"private.md","lastUpdated":null}'),r={name:"private.md"},e=t('<h1 id="Internals" tabindex="-1">Internals <a class="header-anchor" href="#Internals" aria-label="Permalink to &quot;Internals {#Internals}&quot;">​</a></h1><p>Documentation for internal functions.</p><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="TruncatedMVN.mvnrnd!-Tuple{AbstractArray, AbstractArray, TruncatedMVNormal, Vararg{AbstractArray, 4}}" href="#TruncatedMVN.mvnrnd!-Tuple{AbstractArray, AbstractArray, TruncatedMVNormal, Vararg{AbstractArray, 4}}">#</a> <b><u>TruncatedMVN.mvnrnd!</u></b> — <i>Method</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">mvnrnd!</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(z</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, logpr</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, d</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TruncatedMVNormal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, mu</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, L</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, lb</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ub</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Generates samples from a normal distribution. Mutates <code>z</code> and <code>logpr</code> which both an array and a vector with type filled with <code>0.0</code> of dimensions D x n and n respectively.</p><p><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/300a8b732da3fa388194298bd84ef95bfbd2aae0/src/TruncatedMVN.jl#L148-L152" target="_blank" rel="noreferrer">source</a></p></div><br>',4),n=[e];function l(h,d,p,k,o,c){return i(),s("div",null,n)}const g=a(r,[["render",l]]);export{b as __pageData,g as default};
