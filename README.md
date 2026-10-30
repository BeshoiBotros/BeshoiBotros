<div align="center">

# Beshoy Botros

### Backend-focused Full-Stack Engineer · Django · Odoo · Next.js

<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=18&pause=1200&color=2F81F7&center=true&vCenter=true&width=620&lines=Multi-tenant+SaaS+platforms+with+Django+%26+Odoo;Real-time+systems+with+WebSockets+%26+Channels;Offline-first+desktop+apps+with+Electron;APIs+that+schools%2C+clinics+and+companies+run+on" alt="Typing SVG" /></a>

<p>
  <a href="https://www.linkedin.com/in/beshoy-botros-b18350232/"><img src="https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn" /></a>
  <a href="https://twitter.com/BeshoiBotros"><img src="https://img.shields.io/badge/X-000000?style=for-the-badge&logo=x&logoColor=white" alt="X" /></a>
  <a href="https://x-learn.site"><img src="https://img.shields.io/badge/Live-XLearn-2F81F7?style=for-the-badge&logo=googlechrome&logoColor=white" alt="XLearn live" /></a>
</p>

</div>

---

## 👨‍💻 About me

I'm a software engineer from **Cairo, Egypt** who builds the systems that organizations actually run on: learning platforms, school and university management, clinic software, ERPs and legal platforms.

My home is the **backend**. I design data models, multi-tenant architectures, REST and real-time APIs, and background processing. I also ship the **front end** (Next.js, React, Angular) and the **infrastructure** (Docker, Traefik, Jenkins) that puts it in production.

- 🏢 **Multi-tenancy:** schema-per-tenant SaaS with `django-tenants`, and multi-tenant Odoo 17 behind Traefik
- ⚡ **Real-time:** Django Channels, Redis and WebSockets for live proctoring, notifications and chat
- 🤖 **AI in production:** async LLM summaries with Ollama and Celery
- 🌍 **Bilingual products:** full Arabic / English with RTL support across web and desktop
- 🎓 Trained at **ITI** (Information Technology Institute): full-stack and DevOps tracks

---

## 🚀 Featured work

> 🔒 marks private or client repositories. Details are shared here, but the code isn't public.

### 🎓 XLearn: Multi-Tenant Learning Management Platform &nbsp;[![Live](https://img.shields.io/badge/live-x--learn.site-2F81F7?style=flat-square)](https://x-learn.site)

A SaaS LMS where every institution gets its own isolated workspace on its own domain.

- **Schema-per-tenant isolation** with `django-tenants`, plus per-tenant roles and permission groups
- **Courses and content** across modules, with video, video links, books and PowerPoint, modeled polymorphically
- **Assessment engine:** question banks, quizzes and assignments, attempts, per-question marking and submissions review
- **Academic structure** for departments, semesters, enrollments and attendance sessions, all with soft-delete across the domain
- **Encrypted file storage** for course materials, **Celery** background jobs and OpenAPI docs via **drf-spectacular**
- **Next.js dashboard** for admins, instructors and students, with analytics charts and Arabic / English i18n

`Django` `DRF` `django-tenants` `PostgreSQL` `Celery` `Docker` `Next.js` `TypeScript` `Tailwind` `Recharts` · 🔒 backend & web app · [Angular client](https://github.com/BeshoiBotros/xlearn-angular)

---

### 🏥 Clinic Management System: Offline Desktop App &nbsp;🔒

A cross-platform desktop application for running a clinic day to day, with no internet connection required.

- **Patients, appointments, a live waiting room and billing** in one app, with a dashboard of clinic metrics
- **Encrypted local database** (SQLite with ciphers), so patient data stays on the clinic's machine
- **Licensing bound to the device's hardware ID**, plus a first-run setup flow and secure login (bcrypt and JWT)
- **PDF export** for invoices and reports, and a bilingual Arabic / English user guide

`Electron` `Next.js` `React` `TypeScript` `SQLite (encrypted)` `Tailwind` `Recharts`

---

### 🏗️ Multi-Tenant Odoo 17 SaaS ERP &nbsp;[![Repo](https://img.shields.io/badge/repo-ERP-181717?style=flat-square&logo=github)](https://github.com/BeshoiBotros/ERP)

A production-style platform that hosts many Odoo companies on one stack, each on its own subdomain.

- **Traefik v3** edge with wildcard SSL, dynamic subdomain routing, rate limiting and WebSocket proxying
- Network-isolated services, tenant provisioning scripts, and backup and scaling strategy, all documented in `ARCHITECTURE.md`
- **Custom accounting addons:** intercompany accounting (including SO/PO flows), daily and monthly disbursements, and a discount manager

`Odoo 17` `Python` `PostgreSQL` `Docker Compose` `Traefik`

---

### 🏫 Didymus: School Management Platform &nbsp;🔒

A backend for running a school: students, academic structure, attendance, grades and an admin dashboard.

- **Real-time notifications** over Django Channels and Redis, with JWT-authenticated WebSockets
- **Full audit history** of records via `django-simple-history`
- Containerized with Docker and served with Daphne in front of PostgreSQL

`Django` `DRF` `Channels` `Redis` `PostgreSQL` `Docker` · [earlier public version](https://github.com/BeshoiBotros/didymus_backend)

---

### 🩺 AI Consultation System &nbsp;[![Repo](https://img.shields.io/badge/repo-backend-181717?style=flat-square&logo=github)](https://github.com/BeshoiBotros/consultation_system_backend)

A patient and consultation API that writes **AI-generated consultation summaries** with a local LLM.

- LLM inference runs on **Ollama** inside **Celery** workers, so requests never block
- JWT auth, filtering, Redis caching and a one-command Docker Compose setup

`Django` `DRF` `PostgreSQL` `Celery` `Redis` `Ollama` `Docker` · [React client](https://github.com/BeshoiBotros/consultation_frontend)

---

### 🛡️ ProctorIQ: Online Exam Proctoring &nbsp;[![Repo](https://img.shields.io/badge/repo-backend-181717?style=flat-square&logo=github)](https://github.com/BeshoiBotros/ProctorIQ)

An exam platform with **live proctoring over WebSockets**, exam management and subscriptions.

`Django Channels` `Daphne` `DRF` `React` `TypeScript` · [frontend](https://github.com/BeshoiBotros/GP-Proctor-IQ)

---

## 🧩 More projects

| Project | Description | Stack |
|---|---|---|
| **Baccalaureate** 🔒 | Backend for school admission forms, with schools, applications, JWT roles and an admin bootstrap | NestJS · TypeORM · PostgreSQL · Docker |
| **QuantaFlow** 🔒 | Bilingual (EN/AR, RTL) software-agency website with a 3D hero, case studies, insights and a lead form | React 19 · Three.js · Framer Motion · Tailwind v4 |
| **University Portal** 🔒 | University portal backend with rich-text content and data import/export | Django · DRF · PostgreSQL |
| **KPI Analyzer** 🔒 | Desktop tool that automates network KPI checks from exported telecom reports | Python · pandas · Tkinter |
| **[X-Law](https://github.com/BeshoiBotros/api-X-Law-v2)** | Legal platform API for organizations, cases, subscriptions and real-time chat | Django · DRF · Channels |
| **[Apk-Automation](https://github.com/BeshoiBotros/Apk-Automation)** | Upload APKs and automate mobile app testing with UI-hierarchy inspection | Django · Appium · Docker |
| **[Jenkins Shared Library](https://github.com/BeshoiBotros/jenkins-shared-lib)** | Reusable CI/CD pipeline steps for Jenkins and Azure Pipelines | Groovy · Jenkins · Docker |

---

## 🛠️ Tech stack

<table>
  <tr>
    <td><b>Backend</b></td>
    <td><img src="https://skillicons.dev/icons?i=python,django,fastapi,nestjs,nodejs&theme=dark" /></td>
  </tr>
  <tr>
    <td><b>Frontend</b></td>
    <td><img src="https://skillicons.dev/icons?i=nextjs,react,angular,ts,tailwind,threejs&theme=dark" /></td>
  </tr>
  <tr>
    <td><b>Data & Messaging</b></td>
    <td><img src="https://skillicons.dev/icons?i=postgres,mysql,redis,sqlite&theme=dark" /></td>
  </tr>
  <tr>
    <td><b>DevOps</b></td>
    <td><img src="https://skillicons.dev/icons?i=docker,jenkins,azure,linux,git&theme=dark" /></td>
  </tr>
  <tr>
    <td><b>Desktop</b></td>
    <td><img src="https://skillicons.dev/icons?i=electron&theme=dark" /></td>
  </tr>
</table>

Also: **Odoo 17**, **Celery**, **Django Channels**, **django-tenants**, **Traefik**, **TypeORM**, **Ollama**, **Appium**, **pandas**

---

## 📊 GitHub activity

<div align="center">
  <img height="165" src="https://github-readme-stats.vercel.app/api?username=BeshoiBotros&show_icons=true&hide_border=true&count_private=true&include_all_commits=true&theme=transparent&title_color=2F81F7&icon_color=2F81F7" alt="GitHub stats" />
  <img height="165" src="https://github-readme-stats.vercel.app/api/top-langs/?username=BeshoiBotros&layout=compact&hide_border=true&langs_count=8&theme=transparent&title_color=2F81F7" alt="Top languages" />
  <br/>
  <img src="https://streak-stats.demolab.com?user=BeshoiBotros&hide_border=true&theme=transparent&ring=2F81F7&fire=2F81F7&currStreakLabel=2F81F7" alt="GitHub streak" />
</div>

---

<div align="center">

### 📫 Let's work together

I'm open to backend and full-stack roles and to freelance projects.<br/>
The fastest way to reach me is **[LinkedIn](https://www.linkedin.com/in/beshoy-botros-b18350232/)**.

</div>
