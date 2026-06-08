/* eslint-disable */
// @ts-nocheck
import { Route as rootRouteImport } from './routes/__root'
import { Route as UploadRouteImport } from './routes/upload'
import { Route as TrainingRouteImport } from './routes/training'
import { Route as SyntheticRouteImport } from './routes/synthetic'
import { Route as StudioRouteImport } from './routes/studio'
import { Route as SitemapDotxmlRouteImport } from './routes/sitemap[.]xml'
import { Route as SignupRouteImport } from './routes/signup'
import { Route as SigninRouteImport } from './routes/signin'
import { Route as ResetPasswordRouteImport } from './routes/reset-password'
import { Route as ResearchRouteImport } from './routes/research'
import { Route as PricingRouteImport } from './routes/pricing'
import { Route as PlaygroundRouteImport } from './routes/playground'
import { Route as MneRouteImport } from './routes/mne'
import { Route as ModelsRouteImport } from './routes/models'
import { Route as ExperimentsRouteImport } from './routes/experiments'
import { Route as EmbeddingsRouteImport } from './routes/embeddings'
import { Route as Eeg2imageRouteImport } from './routes/eeg2image'
import { Route as DevelopersRouteImport } from './routes/developers'
import { Route as DatasetsRouteImport } from './routes/datasets'
import { Route as ArchitectureRouteImport } from './routes/architecture'
import { Route as AboutRouteImport } from './routes/about'
import { Route as AuthenticatedRouteRouteImport } from './routes/_authenticated/route'
import { Route as IndexRouteImport } from './routes/index'
import { Route as AuthenticatedDashboardRouteImport } from './routes/_authenticated/dashboard'
import { Route as AuthenticatedDashboardIndexRouteImport } from './routes/_authenticated/dashboard.index'
import { Route as ApiEegUploadRouteImport } from './routes/api/eeg/upload'
import { Route as AuthenticatedDashboardAnalysesRouteImport } from './routes/_authenticated/dashboard.analyses'
import { Route as AuthenticatedDashboardResearcherRouteImport } from './routes/_authenticated/dashboard.researcher'
import { Route as AuthenticatedDashboardIndividualRouteImport } from './routes/_authenticated/dashboard.individual'
import { Route as AuthenticatedDashboardEnterpriseRouteImport } from './routes/_authenticated/dashboard.enterprise'

const UploadRoute = UploadRouteImport.update({ id: '/upload', path: '/upload', getParentRoute: () => rootRouteImport } as any)
const TrainingRoute = TrainingRouteImport.update({ id: '/training', path: '/training', getParentRoute: () => rootRouteImport } as any)
const SyntheticRoute = SyntheticRouteImport.update({ id: '/synthetic', path: '/synthetic', getParentRoute: () => rootRouteImport } as any)
const StudioRoute = StudioRouteImport.update({ id: '/studio', path: '/studio', getParentRoute: () => rootRouteImport } as any)
const SitemapDotxmlRoute = SitemapDotxmlRouteImport.update({ id: '/sitemap.xml', path: '/sitemap.xml', getParentRoute: () => rootRouteImport } as any)
const SignupRoute = SignupRouteImport.update({ id: '/signup', path: '/signup', getParentRoute: () => rootRouteImport } as any)
const SigninRoute = SigninRouteImport.update({ id: '/signin', path: '/signin', getParentRoute: () => rootRouteImport } as any)
const ResetPasswordRoute = ResetPasswordRouteImport.update({ id: '/reset-password', path: '/reset-password', getParentRoute: () => rootRouteImport } as any)
const ResearchRoute = ResearchRouteImport.update({ id: '/research', path: '/research', getParentRoute: () => rootRouteImport } as any)
const PricingRoute = PricingRouteImport.update({ id: '/pricing', path: '/pricing', getParentRoute: () => rootRouteImport } as any)
const PlaygroundRoute = PlaygroundRouteImport.update({ id: '/playground', path: '/playground', getParentRoute: () => rootRouteImport } as any)
const MneRoute = MneRouteImport.update({ id: '/mne', path: '/mne', getParentRoute: () => rootRouteImport } as any)
const ModelsRoute = ModelsRouteImport.update({ id: '/models', path: '/models', getParentRoute: () => rootRouteImport } as any)
const ExperimentsRoute = ExperimentsRouteImport.update({ id: '/experiments', path: '/experiments', getParentRoute: () => rootRouteImport } as any)
const EmbeddingsRoute = EmbeddingsRouteImport.update({ id: '/embeddings', path: '/embeddings', getParentRoute: () => rootRouteImport } as any)
const Eeg2imageRoute = Eeg2imageRouteImport.update({ id: '/eeg2image', path: '/eeg2image', getParentRoute: () => rootRouteImport } as any)
const DevelopersRoute = DevelopersRouteImport.update({ id: '/developers', path: '/developers', getParentRoute: () => rootRouteImport } as any)
const DatasetsRoute = DatasetsRouteImport.update({ id: '/datasets', path: '/datasets', getParentRoute: () => rootRouteImport } as any)
const ArchitectureRoute = ArchitectureRouteImport.update({ id: '/architecture', path: '/architecture', getParentRoute: () => rootRouteImport } as any)
const AboutRoute = AboutRouteImport.update({ id: '/about', path: '/about', getParentRoute: () => rootRouteImport } as any)
const IndexRoute = IndexRouteImport.update({ id: '/', path: '/', getParentRoute: () => rootRouteImport } as any)
const AuthenticatedRouteRoute = AuthenticatedRouteRouteImport.update({ id: '/_authenticated', getParentRoute: () => rootRouteImport } as any)
const ApiEegUploadRoute = ApiEegUploadRouteImport.update({ id: '/api/eeg/upload', path: '/api/eeg/upload', getParentRoute: () => rootRouteImport } as any)
const AuthenticatedDashboardRoute = AuthenticatedDashboardRouteImport.update({ id: '/dashboard', path: '/dashboard', getParentRoute: () => AuthenticatedRouteRoute } as any)
const AuthenticatedDashboardIndexRoute = AuthenticatedDashboardIndexRouteImport.update({ id: '/', path: '/', getParentRoute: () => AuthenticatedDashboardRoute } as any)
const AuthenticatedDashboardAnalysesRoute = AuthenticatedDashboardAnalysesRouteImport.update({ id: '/analyses', path: '/analyses', getParentRoute: () => AuthenticatedDashboardRoute } as any)
const AuthenticatedDashboardResearcherRoute = AuthenticatedDashboardResearcherRouteImport.update({ id: '/researcher', path: '/researcher', getParentRoute: () => AuthenticatedDashboardRoute } as any)
const AuthenticatedDashboardIndividualRoute = AuthenticatedDashboardIndividualRouteImport.update({ id: '/individual', path: '/individual', getParentRoute: () => AuthenticatedDashboardRoute } as any)
const AuthenticatedDashboardEnterpriseRoute = AuthenticatedDashboardEnterpriseRouteImport.update({ id: '/enterprise', path: '/enterprise', getParentRoute: () => AuthenticatedDashboardRoute } as any)

export interface FileRoutesByFullPath {
  '/': typeof IndexRoute
  '/about': typeof AboutRoute
  '/architecture': typeof ArchitectureRoute
  '/datasets': typeof DatasetsRoute
  '/developers': typeof DevelopersRoute
  '/eeg2image': typeof Eeg2imageRoute
  '/embeddings': typeof EmbeddingsRoute
  '/experiments': typeof ExperimentsRoute
  '/mne': typeof MneRoute
  '/models': typeof ModelsRoute
  '/playground': typeof PlaygroundRoute
  '/pricing': typeof PricingRoute
  '/research': typeof ResearchRoute
  '/reset-password': typeof ResetPasswordRoute
  '/signin': typeof SigninRoute
  '/signup': typeof SignupRoute
  '/sitemap.xml': typeof SitemapDotxmlRoute
  '/studio': typeof StudioRoute
  '/synthetic': typeof SyntheticRoute
  '/training': typeof TrainingRoute
  '/upload': typeof UploadRoute
  '/api/eeg/upload': typeof ApiEegUploadRoute
  '/_authenticated': typeof AuthenticatedRouteRoute
  '/dashboard': typeof AuthenticatedDashboardRoute
  '/dashboard/': typeof AuthenticatedDashboardIndexRoute
  '/dashboard/analyses': typeof AuthenticatedDashboardAnalysesRoute
  '/dashboard/enterprise': typeof AuthenticatedDashboardEnterpriseRoute
  '/dashboard/individual': typeof AuthenticatedDashboardIndividualRoute
  '/dashboard/researcher': typeof AuthenticatedDashboardResearcherRoute
}

export interface FileRoutesByTo {
  '/': typeof IndexRoute
  '/about': typeof AboutRoute
  '/architecture': typeof ArchitectureRoute
  '/datasets': typeof DatasetsRoute
  '/developers': typeof DevelopersRoute
  '/eeg2image': typeof Eeg2imageRoute
  '/embeddings': typeof EmbeddingsRoute
  '/experiments': typeof ExperimentsRoute
  '/mne': typeof MneRoute
  '/models': typeof ModelsRoute
  '/playground': typeof PlaygroundRoute
  '/pricing': typeof PricingRoute
  '/research': typeof ResearchRoute
  '/reset-password': typeof ResetPasswordRoute
  '/signin': typeof SigninRoute
  '/signup': typeof SignupRoute
  '/sitemap.xml': typeof SitemapDotxmlRoute
  '/studio': typeof StudioRoute
  '/synthetic': typeof SyntheticRoute
  '/training': typeof TrainingRoute
  '/upload': typeof UploadRoute
  '/api/eeg/upload': typeof ApiEegUploadRoute
  '/dashboard': typeof AuthenticatedDashboardIndexRoute
  '/dashboard/analyses': typeof AuthenticatedDashboardAnalysesRoute
  '/dashboard/enterprise': typeof AuthenticatedDashboardEnterpriseRoute
  '/dashboard/individual': typeof AuthenticatedDashboardIndividualRoute
  '/dashboard/researcher': typeof AuthenticatedDashboardResearcherRoute
}

export interface FileRoutesById {
  __root__: typeof rootRouteImport
  '/': typeof IndexRoute
  '/about': typeof AboutRoute
  '/architecture': typeof ArchitectureRoute
  '/datasets': typeof DatasetsRoute
  '/developers': typeof DevelopersRoute
  '/eeg2image': typeof Eeg2imageRoute
  '/embeddings': typeof EmbeddingsRoute
  '/experiments': typeof ExperimentsRoute
  '/mne': typeof MneRoute
  '/models': typeof ModelsRoute
  '/playground': typeof PlaygroundRoute
  '/pricing': typeof PricingRoute
  '/research': typeof ResearchRoute
  '/reset-password': typeof ResetPasswordRoute
  '/signin': typeof SigninRoute
  '/signup': typeof SignupRoute
  '/sitemap.xml': typeof SitemapDotxmlRoute
  '/studio': typeof StudioRoute
  '/synthetic': typeof SyntheticRoute
  '/training': typeof TrainingRoute
  '/upload': typeof UploadRoute
  '/api/eeg/upload': typeof ApiEegUploadRoute
  '/_authenticated': typeof AuthenticatedRouteRoute
  '/_authenticated/dashboard': typeof AuthenticatedDashboardRoute
  '/_authenticated/dashboard/': typeof AuthenticatedDashboardIndexRoute
  '/_authenticated/dashboard/analyses': typeof AuthenticatedDashboardAnalysesRoute
  '/_authenticated/dashboard/enterprise': typeof AuthenticatedDashboardEnterpriseRoute
  '/_authenticated/dashboard/individual': typeof AuthenticatedDashboardIndividualRoute
  '/_authenticated/dashboard/researcher': typeof AuthenticatedDashboardResearcherRoute
}

export interface FileRouteTypes {
  fileRoutesByFullPath: FileRoutesByFullPath
  fileRoutesByTo: FileRoutesByTo
  fileRoutesById: FileRoutesById
}

export interface RootRouteChildren {
  IndexRoute: typeof IndexRoute
  AboutRoute: typeof AboutRoute
  ArchitectureRoute: typeof ArchitectureRoute
  DatasetsRoute: typeof DatasetsRoute
  DevelopersRoute: typeof DevelopersRoute
  Eeg2imageRoute: typeof Eeg2imageRoute
  EmbeddingsRoute: typeof EmbeddingsRoute
  ExperimentsRoute: typeof ExperimentsRoute
  MneRoute: typeof MneRoute
  ModelsRoute: typeof ModelsRoute
  PlaygroundRoute: typeof PlaygroundRoute
  PricingRoute: typeof PricingRoute
  ResearchRoute: typeof ResearchRoute
  ResetPasswordRoute: typeof ResetPasswordRoute
  SigninRoute: typeof SigninRoute
  SignupRoute: typeof SignupRoute
  SitemapDotxmlRoute: typeof SitemapDotxmlRoute
  StudioRoute: typeof StudioRoute
  SyntheticRoute: typeof SyntheticRoute
  TrainingRoute: typeof TrainingRoute
  UploadRoute: typeof UploadRoute
  ApiEegUploadRoute: typeof ApiEegUploadRoute
  AuthenticatedRouteRoute: typeof AuthenticatedRouteRoute
}

const rootRouteChildren: RootRouteChildren = {
  IndexRoute, AboutRoute, ArchitectureRoute, DatasetsRoute,
  DevelopersRoute, Eeg2imageRoute, EmbeddingsRoute, ExperimentsRoute,
  MneRoute, ModelsRoute, PlaygroundRoute, PricingRoute,
  ResearchRoute, ResetPasswordRoute, SigninRoute, SignupRoute,
  SitemapDotxmlRoute, StudioRoute, SyntheticRoute, TrainingRoute,
  UploadRoute, ApiEegUploadRoute, AuthenticatedRouteRoute,
}

export const routeTree = rootRouteImport
  ._addFileChildren(rootRouteChildren)
  ._addFileTypes<FileRouteTypes>()

export interface AuthenticatedDashboardRouteChildren {
  AuthenticatedDashboardIndexRoute: typeof AuthenticatedDashboardIndexRoute
  AuthenticatedDashboardAnalysesRoute: typeof AuthenticatedDashboardAnalysesRoute
  AuthenticatedDashboardEnterpriseRoute: typeof AuthenticatedDashboardEnterpriseRoute
  AuthenticatedDashboardIndividualRoute: typeof AuthenticatedDashboardIndividualRoute
  AuthenticatedDashboardResearcherRoute: typeof AuthenticatedDashboardResearcherRoute
}

const AuthenticatedDashboardRouteChildren: AuthenticatedDashboardRouteChildren = {
  AuthenticatedDashboardIndexRoute,
  AuthenticatedDashboardAnalysesRoute,
  AuthenticatedDashboardEnterpriseRoute,
  AuthenticatedDashboardIndividualRoute,
  AuthenticatedDashboardResearcherRoute,
}

export const AuthenticatedDashboardRouteWithChildren =
  AuthenticatedDashboardRoute._addFileChildren(AuthenticatedDashboardRouteChildren)
