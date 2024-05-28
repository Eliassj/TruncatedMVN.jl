import{_ as a,c as i,o as t,a6 as s}from"./chunks/framework.Cdb85ycF.js";const b=JSON.parse('{"title":"TruncatedMVN","description":"","frontmatter":{"layout":"home","hero":{"name":"TruncatedMVN.jl","tagline":"Truncated multivariate normal distribution with exact sampling"}},"headers":[],"relativePath":"index.md","filePath":"index.md","lastUpdated":null}'),e={name:"index.md"},r=s('<h1 id="TruncatedMVN" tabindex="-1">TruncatedMVN <a class="header-anchor" href="#TruncatedMVN" aria-label="Permalink to &quot;TruncatedMVN {#TruncatedMVN}&quot;">​</a></h1><p>Julia reimplementation of a truncated multivariate normal distribution with perfect sampling.</p><p>Distribution and sampling method by [<a href="/Eliassj.github.io/TruncatedMVN.jl/dev/index#Botev_2017">1</a>].</p><p>Code based on MATLAB implementation by Zdravko Botev and Python implementation by Paul Brunzema. These may be found here: <a href="https://mathworks.com/matlabcentral/fileexchange/53792-truncated-multivariate-normal-generator" target="_blank" rel="noreferrer">MATLAB by Botev</a>, <a href="https://github.com/brunzema/truncated-mvn-sampler" target="_blank" rel="noreferrer">Python by Brunzema</a></p><p>Exports 1 struct + its constructor and 1 function:</p><ul><li><p><a href="/Eliassj.github.io/TruncatedMVN.jl/dev/index#TruncatedMVN.TruncatedMVNormal"><code>TruncatedMVN.TruncatedMVNormal</code></a>: Struct and constructor for initializing the distribution.</p></li><li><p><a href="/Eliassj.github.io/TruncatedMVN.jl/dev/index#TruncatedMVN.sample"><code>TruncatedMVN.sample</code></a>: Exact sampling from the distribution.</p></li></ul><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="TruncatedMVN.TruncatedMVNormal" href="#TruncatedMVN.TruncatedMVNormal">#</a> <b><u>TruncatedMVN.TruncatedMVNormal</u></b> — <i>Type</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">TruncatedMVNormal{S</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{&lt;:AbstractFloat}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractVector{&lt;:AbstractFloat}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,U</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Integer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,V</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractFloat</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,P</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractVector{&lt;:Integer}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Truncated multivariate normal distribution with minimax tilting-based sampling.</p><p><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/117b0c11382f708df8e26f97118224230cdda055/src/TruncatedMVN.jl#L19-L24" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T&lt;:(AbstractVector{&lt;:AbstractFloat}), S&lt;:(AbstractArray{&lt;:AbstractFloat})}" href="#TruncatedMVN.TruncatedMVNormal-Union{Tuple{S}, Tuple{T}, Tuple{T, S, T, T}} where {T&lt;:(AbstractVector{&lt;:AbstractFloat}), S&lt;:(AbstractArray{&lt;:AbstractFloat})}">#</a> <b><u>TruncatedMVN.TruncatedMVNormal</u></b> — <i>Method</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;"> TruncatedMVNormal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(mu</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">T</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, cov</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">S</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, lb</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">T</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, ub</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">T</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">) </span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">where</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;"> {T</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractVector{&lt;:AbstractFloat}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">,S</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">&lt;:</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">AbstractArray{&lt;:AbstractFloat}</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">}</span></span></code></pre></div><p>Inner constructor of the <a href="/Eliassj.github.io/TruncatedMVN.jl/dev/index#TruncatedMVN.TruncatedMVNormal"><code>TruncatedMVN.TruncatedMVNormal</code></a> distribution.</p><p>Generates a truncated multivariate normal distribution which may be accurately sampled from using <a href="/Eliassj.github.io/TruncatedMVN.jl/dev/index#TruncatedMVN.sample"><code>TruncatedMVN.sample</code></a>.</p><p><strong>Arguments</strong></p><ul><li><p><code>mu::T</code>: D-dimensional vector of means.</p></li><li><p><code>cov::S</code>: DxD-dimensional covariance matrix.</p></li><li><p><code>lb::T</code>: D-dimensional vector of lower bounds.</p></li><li><p><code>ub::T</code>: D-dimensional vector of upper bounds.</p></li></ul><p>Bounds may be <code>-Inf</code>/<code>Inf</code>.</p><p><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/117b0c11382f708df8e26f97118224230cdda055/src/TruncatedMVN.jl#L41-L57" target="_blank" rel="noreferrer">source</a></p></div><br><div style="border-width:1px;border-style:solid;border-color:black;padding:1em;border-radius:25px;"><a id="TruncatedMVN.sample" href="#TruncatedMVN.sample">#</a> <b><u>TruncatedMVN.sample</u></b> — <i>Function</i>. <div class="language-julia vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang">julia</span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">sample</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">(d</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">TruncatedMVNormal</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, n</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Integer</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">, max_iter</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">::</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">Integer</span><span style="--shiki-light:#D73A49;--shiki-dark:#F97583;">=</span><span style="--shiki-light:#005CC5;--shiki-dark:#79B8FF;">10000</span><span style="--shiki-light:#24292E;--shiki-dark:#E1E4E8;">)</span></span></code></pre></div><p>Sample <code>n</code> samples from the distribution <code>d</code>.</p><p>Returns an D x n <code>Matrix</code> of samples where D is the dimension of the distribution <code>d</code>.</p><p><a href="https://github.com/Eliassj/TruncatedMVN.jl/blob/117b0c11382f708df8e26f97118224230cdda055/src/TruncatedMVN.jl#L76-L82" target="_blank" rel="noreferrer">source</a></p></div><br><h1 id="References" tabindex="-1">References <a class="header-anchor" href="#References" aria-label="Permalink to &quot;References {#References}&quot;">​</a></h1><hr><h1 id="bibliography" tabindex="-1">Bibliography <a class="header-anchor" href="#bibliography" aria-label="Permalink to &quot;Bibliography&quot;">​</a></h1><ol><li>Z. I. Botev. <a href="http://dx.doi.org/10.1111/rssb.12162" target="_blank" rel="noreferrer"><em>The normal law under linear restrictions: Simulation and estimation via minimax tilting</em></a>. <a href="https://doi.org/10.1111/rssb.12162" target="_blank" rel="noreferrer">Journal of the Royal Statistical Society. Series B, Statistical methodology <strong>79</strong>, 125–148</a> (2017).</li></ol>',16),n=[r];function l(d,h,o,p,c,k){return t(),i("div",null,n)}const g=a(e,[["render",l]]);export{b as __pageData,g as default};